/**
 * @file    parallel_heat_solver.cpp
 * @author  xkrejc69 <xkrejc69@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2021-05-01
 */

#include "parallel_heat_solver.h"

#include <hdf5.h>

//============================================================================//
//                            *** BEGIN: NOTE ***
//
// Implement methods of your ParallelHeatSolver class here.
// Freely modify any existing code in ***THIS FILE*** as only stubs are provided 
// to allow code to compile.
//
//                             *** END: NOTE ***
//============================================================================//

constexpr int MPI_ROOT_RANK = 0;

constexpr int ERR_INVALID_DECOMPOSITION = 1;

ParallelHeatSolver::ParallelHeatSolver(SimulationProperties &simulationProps, MaterialProperties &materialProps)
    :   BaseHeatSolver (simulationProps, materialProps),
        m_fileHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr)),
        edgeSize(materialProps.GetEdgeSize())
{
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    //// 1) Input file
    
    // Creating EMPTY HDF5 handle using RAII "AutoHandle" type
    //
    // AutoHandle<hid_t> myHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
    //
    // This can be particularly useful when creating handle as class member!
    // Handle CANNOT be assigned using "=" or copy-constructor, yet it can be set
    // using ".Set(/* handle */, /* close/free function */)" as in:
    // myHandle.Set(H5Fopen(...), H5Fclose);
    if(!m_simulationProperties.GetOutputFileName().empty()){
        if(m_simulationProperties.IsUseParallelIO()){
            // 1. Create a property list to open the HDF5 file using MPI-IO in the MPI_COMM_WORLD communicator.
            hid_t accesPList = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(accesPList, MPI_COMM_WORLD, MPI_INFO_NULL);

            // 3. Create a file called with write permission. Use such a flag that overrides existing file.
            hid_t file = H5Fcreate(simulationProps.GetOutputFileName("par").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, accesPList);
            m_fileHandle.Set(file, H5Fclose);

            H5Pclose(accesPList);

            // TODO
            /*
            // 5. Create file space - a 2D matrix [edgeSize][edgeSize]
            //    Create mem space  - a 2D matrix [lRows][nCols] mapped on 1D array lMatrix.
            hsize_t rank = 2;
            hsize_t datasetSize[] = {hsize_t(edgeSize), hsize_t(edgeSize)};
            hsize_t memSize[]     = {hsize_t(lRows), hsize_t(nCols)};

            hid_t filespace = H5Screate_simple(rank, datasetSize, nullptr);
            hid_t memspace  = H5Screate_simple(rank, memSize,     nullptr);
            */
        }
        else if(m_rank == MPI_ROOT_RANK){
            hid_t file = H5Fcreate(simulationProps.GetOutputFileName("par").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            m_fileHandle.Set(file, H5Fclose);
        }
    }

    //// 2) Decomposition

    // Requested domain decomposition can be queried by
    // m_simulationProperties.GetDecompGrid(/* TILES IN X */, /* TILES IN Y */)
    m_simulationProperties.GetDecompGrid(tilesX, tilesY);

    // tile sizes with and without borders
    tileCols = edgeSize / tilesX;
    tileRows = edgeSize / tilesY;
    tileSize = tileCols * tileRows;
    extendedTileCols = tileCols + DOUBLE_OFFSET;
    extendedTileRows = tileRows + DOUBLE_OFFSET;
    extendedTileSize = extendedTileCols * extendedTileRows;

    if(m_rank == MPI_ROOT_RANK && (tileCols < OFFSET || tileRows < OFFSET)){
        std::cerr
            << "Requires a decomposition with tile of at least size [" << OFFSET << ", " << OFFSET << "],"
            << "used [" << tileCols << ", " << tileRows << "]\n" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_DECOMPOSITION);
    }
    
    matrixSize = edgeSize * edgeSize;

    //// 3) Helper arrays for each process
    lTempArray1.resize(extendedTileSize);
    lTempArray2.resize(extendedTileSize);
    lDomainParams.resize(extendedTileSize);
    lDomainMap.resize(extendedTileSize);

    // position info
    atLeftBorder = (m_rank % tilesX) == 0;
    atRightBorder = (m_rank % tilesX) == (tilesX - 1);
    atTopBorder = m_rank < tilesX; // is in first row
    atBottomBorder = m_rank >= (m_size - tilesX); // one of the last [number of tiles in X] tiles

    // indices for tile without borders
    tileStartX = OFFSET;
    tileEndX = extendedTileCols - OFFSET;
    tileStartY = OFFSET;
    tileEndY = extendedTileRows - OFFSET;
    if(atLeftBorder)
        tileStartX += OFFSET;
    if(atRightBorder)
        tileEndX -= OFFSET;
    if(atTopBorder)
        tileStartY += OFFSET;
    if(atBottomBorder)
        tileEndY -= OFFSET;

    // initialize types
	constexpr int ndims = 2;
	const int size[ndims] = {extendedTileRows, extendedTileCols};
	const int subsize[ndims] = {tileRows, tileCols};
	int start[ndims] = {OFFSET, OFFSET};

    // worker tiles
	MPI_Type_create_subarray(ndims, size, subsize, start, MPI_ORDER_C, MPI_FLOAT, &TYPE_WORKER_TILE_FLOAT);
	MPI_Type_commit(&TYPE_WORKER_TILE_FLOAT);
	MPI_Type_create_subarray(ndims, size, subsize, start, MPI_ORDER_C, MPI_INT, &TYPE_WORKER_TILE_INT);
	MPI_Type_commit(&TYPE_WORKER_TILE_INT);

    leftBorderSendIdx = OFFSET * extendedTileCols + OFFSET;
	rightBorderRecvIdx = (OFFSET + 1) * extendedTileCols - OFFSET;
	rightBorderSendIdx = (OFFSET + 1) * extendedTileCols - DOUBLE_OFFSET;
	leftBorderRecvIdx = OFFSET * extendedTileCols;
	topBorderSendIdx = OFFSET * extendedTileCols + OFFSET;
	bottomBorderRecvIdx = (OFFSET + tileRows) * extendedTileCols + OFFSET;
    bottomBorderSendIdx = tileRows * extendedTileCols + OFFSET;
	topBorderRecvIdx = OFFSET;

    // worker borders
	MPI_Type_vector(tileRows, OFFSET, extendedTileCols, MPI_FLOAT, &TYPE_TILE_BORDER_LR_FLOAT);
	MPI_Type_commit(&TYPE_TILE_BORDER_LR_FLOAT);
	MPI_Type_vector(OFFSET, tileCols, extendedTileCols, MPI_FLOAT, &TYPE_TILE_BORDER_TB_FLOAT);
	MPI_Type_commit(&TYPE_TILE_BORDER_TB_FLOAT);
	MPI_Type_vector(tileRows, OFFSET, extendedTileCols, MPI_INT, &TYPE_TILE_BORDER_LR_INT);
	MPI_Type_commit(&TYPE_TILE_BORDER_LR_INT);
	MPI_Type_vector(OFFSET, tileCols, extendedTileCols, MPI_INT, &TYPE_TILE_BORDER_TB_INT);
	MPI_Type_commit(&TYPE_TILE_BORDER_TB_INT);

    // root tiles
	if(m_rank == MPI_ROOT_RANK){
		const int rsize[ndims] = {edgeSize, edgeSize};
		const int rsubsize[ndims] = {tileRows, tileCols};
		const int rstart[ndims] = {0, 0};

		MPI_Type_create_subarray(ndims, rsize, rsubsize, rstart, MPI_ORDER_C, MPI_FLOAT, &TYPE_ROOT_TILE_FLOAT_INITIAL);
		MPI_Type_commit(&TYPE_ROOT_TILE_FLOAT_INITIAL);
		MPI_Type_create_resized(TYPE_ROOT_TILE_FLOAT_INITIAL, 0, sizeof(float), &TYPE_ROOT_TILE_FLOAT);
		MPI_Type_commit(&TYPE_ROOT_TILE_FLOAT);

		MPI_Type_create_subarray(ndims, rsize, rsubsize, rstart, MPI_ORDER_C, MPI_INT, &TYPE_ROOT_TILE_INT_INITIAL);
		MPI_Type_commit(&TYPE_ROOT_TILE_INT_INITIAL);
		MPI_Type_create_resized(TYPE_ROOT_TILE_INT_INITIAL, 0, sizeof(int), &TYPE_ROOT_TILE_INT);
		MPI_Type_commit(&TYPE_ROOT_TILE_INT);
	}

    // scatter tiles
    vTileCounts.resize(tilesX * tilesY);
    std::fill(vTileCounts.begin(), vTileCounts.end(), 1);

    vTileDisplacements.resize(tilesX * tilesY);
    int k = 0;
    for(int i = 0; i < tilesY; i++){
        for(int j = 0; j < tilesX; j++){
            vTileDisplacements[k++] = (tileSize * tilesX * i) + (tileCols * j);
        }
    }

    MPI_Scatterv(m_materialProperties.GetInitTemp().data(), vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_FLOAT, lTempArray1.data(), 1, TYPE_WORKER_TILE_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Scatterv(m_materialProperties.GetInitTemp().data(), vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_FLOAT, lTempArray2.data(), 1, TYPE_WORKER_TILE_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Scatterv(m_materialProperties.GetDomainParams().data(), vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_FLOAT, lDomainParams.data(), 1, TYPE_WORKER_TILE_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Scatterv(m_materialProperties.GetDomainMap().data(), vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_INT, lDomainMap.data(), 1, TYPE_WORKER_TILE_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    // neighbour indices
    neighbourLeft = m_rank - 1;
    neighbourRight = m_rank + 1;
    neighbourTop = m_rank - tilesX;
    neighbourBottom = m_rank + tilesX;

    initialBorderExchange();
}

void ParallelHeatSolver::initialBorderExchange(){
    MPI_Request reqs[8 * 3];
    unsigned i = 0;

    unsigned idx;
    int neighbour;

    if(!atLeftBorder){
        neighbour = neighbourLeft;
        idx = leftBorderSendIdx;

        MPI_Isend(&lTempArray1[idx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbour, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Isend(&lDomainParams[idx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbour, TAG_INIT_BORDER_DOMAIN_PARAMS, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Isend(&lDomainMap[idx], 1, TYPE_TILE_BORDER_LR_INT, neighbour, TAG_INIT_BORDER_DOMAIN_MAP, MPI_COMM_WORLD, &reqs[i++]);

        idx = leftBorderRecvIdx;

        MPI_Irecv(&lTempArray1[idx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbour, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Irecv(&lDomainParams[idx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbour, TAG_INIT_BORDER_DOMAIN_PARAMS, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Irecv(&lDomainMap[idx], 1, TYPE_TILE_BORDER_LR_INT, neighbour, TAG_INIT_BORDER_DOMAIN_MAP, MPI_COMM_WORLD, &reqs[i++]);
    }
    if(!atRightBorder){
        neighbour = neighbourRight;
        idx = rightBorderSendIdx;

        MPI_Isend(&lTempArray1[idx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbour, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Isend(&lDomainParams[idx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbour, TAG_INIT_BORDER_DOMAIN_PARAMS, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Isend(&lDomainMap[idx], 1, TYPE_TILE_BORDER_LR_INT, neighbour, TAG_INIT_BORDER_DOMAIN_MAP, MPI_COMM_WORLD, &reqs[i++]);

        idx = rightBorderRecvIdx;

        MPI_Irecv(&lTempArray1[idx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbour, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Irecv(&lDomainParams[idx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbour, TAG_INIT_BORDER_DOMAIN_PARAMS, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Irecv(&lDomainMap[idx], 1, TYPE_TILE_BORDER_LR_INT, neighbour, TAG_INIT_BORDER_DOMAIN_MAP, MPI_COMM_WORLD, &reqs[i++]);
    }
    if(!atTopBorder){
        neighbour = neighbourTop;
        idx = topBorderSendIdx;

        MPI_Isend(&lTempArray1[idx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbour, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Isend(&lDomainParams[idx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbour, TAG_INIT_BORDER_DOMAIN_PARAMS, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Isend(&lDomainMap[idx], 1, TYPE_TILE_BORDER_TB_INT, neighbour, TAG_INIT_BORDER_DOMAIN_MAP, MPI_COMM_WORLD, &reqs[i++]);

        idx = topBorderRecvIdx;

        MPI_Irecv(&lTempArray1[idx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbour, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Irecv(&lDomainParams[idx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbour, TAG_INIT_BORDER_DOMAIN_PARAMS, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Irecv(&lDomainMap[idx], 1, TYPE_TILE_BORDER_TB_INT, neighbour, TAG_INIT_BORDER_DOMAIN_MAP, MPI_COMM_WORLD, &reqs[i++]);
    }
    if(!atBottomBorder){
        neighbour = neighbourBottom;
        idx = bottomBorderSendIdx;

        MPI_Isend(&lTempArray1[idx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbour, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Isend(&lDomainParams[idx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbour, TAG_INIT_BORDER_DOMAIN_PARAMS, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Isend(&lDomainMap[idx], 1, TYPE_TILE_BORDER_TB_INT, neighbour, TAG_INIT_BORDER_DOMAIN_MAP, MPI_COMM_WORLD, &reqs[i++]);

        idx = bottomBorderRecvIdx;

        MPI_Irecv(&lTempArray1[idx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbour, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Irecv(&lDomainParams[idx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbour, TAG_INIT_BORDER_DOMAIN_PARAMS, MPI_COMM_WORLD, &reqs[i++]);
        MPI_Irecv(&lDomainMap[idx], 1, TYPE_TILE_BORDER_TB_INT, neighbour, TAG_INIT_BORDER_DOMAIN_MAP, MPI_COMM_WORLD, &reqs[i++]);
    }

    MPI_Waitall(i, reqs, MPI_STATUSES_IGNORE);
}

ParallelHeatSolver::~ParallelHeatSolver(){
}

void printMat(float* mat, int rows, int cols){ // TODO rm
	for(int i = 0; i < rows; i ++){
		for(int j = 0; j < cols; j ++){
			const int idx = (i * cols) + j;
			std::cout << mat[idx] << "\t";
		}
		std::cout << std::endl;
	}
}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float>>& outResult){
    float* workTempArrays[2] = {lTempArray1.data(), lTempArray2.data()};

    if(m_rank == 0){
        std::cout <<
            "edgeSize: " << edgeSize <<
            "\ttilesX: " << tilesX << "\ttilesY: " << tilesY << 
            "\ttileCols: " << tileCols << "\ttileRows: " << tileRows << std::endl; 
    }
    /*
    if(m_rank == 0){
        std::cout <<
            (atLeftBorder ? "L" : "-") << "\t" <<
            (atRightBorder ? "R" : "-") << "\t" <<
            (atTopBorder ? "T" : "-") << "\t" <<
            (atBottomBorder ? "B" : "-") << std::endl;
    }
    */

    const int lenX = tileEndX - tileStartX;
    const int lenY = tileEndY - tileStartY;

    if(m_rank == 0){
        std::cout <<
            "rank" << m_rank << "\t" <<
            "tileStartX: " << tileStartX << "\t" <<
            "tileEndX: " << tileEndX << "\t" <<
            "tileStartY: " << tileStartY << "\t" <<
            "tileEndY: " << tileEndY << "\t" <<
            "lenX: " << lenX << "\t" <<
            "lenY: " << lenY << std::endl;
    }
    if(m_rank == 1){
        std::cout <<
            "rank" << m_rank << "\t" <<
            "tileStartX: " << tileStartX << "\t" <<
            "tileEndX: " << tileEndX << "\t" <<
            "tileStartY: " << tileStartY << "\t" <<
            "tileEndY: " << tileEndY << "\t" <<
            "lenX: " << lenX << "\t" <<
            "lenY: " << lenY << std::endl;
    }

    // UpdateTile(...) method can be used to evaluate heat equation over 2D tile
    //                 in parallel (using OpenMP).
    // NOTE: This method might be inefficient when used for small tiles such as 
    //       2xN or Nx2 (these might arise at edges of the tile)
    //       In this case ComputePoint may be called directly in loop.
    for(size_t iter = 0; iter < m_simulationProperties.GetNumIterations(); iter++){
        UpdateTile(
            workTempArrays[0], workTempArrays[1], lDomainParams.data(), lDomainMap.data(),
            tileStartX, tileStartY, lenX, lenY, extendedTileCols,
            m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp()
        );

        MPI_Request reqs[8];
        unsigned i = 0;
        if(!atLeftBorder){
            MPI_Isend(&workTempArrays[1][leftBorderSendIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourLeft, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
            MPI_Irecv(&workTempArrays[1][leftBorderRecvIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourLeft, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        }
        if(!atRightBorder){
            MPI_Isend(&workTempArrays[1][rightBorderSendIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourRight, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
            MPI_Irecv(&workTempArrays[1][rightBorderRecvIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourRight, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        }
        if(!atTopBorder){
            MPI_Isend(&workTempArrays[1][topBorderSendIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourTop, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
            MPI_Irecv(&workTempArrays[1][topBorderRecvIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourTop, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        }
        if(!atBottomBorder){
            MPI_Isend(&workTempArrays[1][bottomBorderSendIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourBottom, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
            MPI_Irecv(&workTempArrays[1][bottomBorderRecvIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourBottom, TAG_INIT_BORDER_TEMP, MPI_COMM_WORLD, &reqs[i++]);
        }
        MPI_Waitall(i, reqs, MPI_STATUSES_IGNORE);

        std::swap(workTempArrays[0], workTempArrays[1]);
    }

    MPI_Gatherv(
        workTempArrays[0], 1, TYPE_WORKER_TILE_FLOAT, outResult.data(),
        vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_FLOAT,
        MPI_ROOT_RANK, MPI_COMM_WORLD
    );
    
    // ShouldPrintProgress(N) returns true if average temperature should be reported
    // by 0th process at Nth time step (using "PrintProgressReport(...)").
    
    // Finally "PrintFinalReport(...)" should be used to print final elapsed time and
    // average temperature in column.
}
