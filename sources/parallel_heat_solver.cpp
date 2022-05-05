/**
 * @file    parallel_heat_solver.cpp
 * @author  xkrejc69 <xkrejc69@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2021-05-05
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
		filespaceHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr)),
		memspaceHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr)),
		xferPListHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr)),
		edgeSize(materialProps.GetEdgeSize())
{
	MPI_Comm_size(MPI_COMM_WORLD, &m_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

	initSimulation();
}

void ParallelHeatSolver::initSimulation(){
	initDecomposition();
	initIO();
	initMemory();
	initTileInfo();
	initTypes();
	initRMA();
	initMiddleColumnInfo();
	initScatterTiles();
	initBorderExchange();
}

void ParallelHeatSolver::initDecomposition(){
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

	matrixSize = edgeSize * edgeSize;

	// size of tile cannot be less than size of border in any dimension
	if(tileCols < OFFSET || tileRows < OFFSET){
		if(m_rank == MPI_ROOT_RANK){
			std::cerr
				<< "Requires a decomposition with tiles of at least size [" << OFFSET << ", " << OFFSET << "], "
				<< "used [" << tileCols << ", " << tileRows << "]\n" << std::endl;
			MPI_Abort(MPI_COMM_WORLD, ERR_INVALID_DECOMPOSITION);
		}
		MPI_Barrier(MPI_COMM_WORLD); // wait for root to print error and abort
	}
}

void ParallelHeatSolver::initIO(){
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
			// Create a property list to open the HDF5 file using MPI-IO in the MPI_COMM_WORLD communicator.
			hid_t accesPList = H5Pcreate(H5P_FILE_ACCESS);
			H5Pset_fapl_mpio(accesPList, MPI_COMM_WORLD, MPI_INFO_NULL);

			// Create a file called with write permission. Use such a flag that overrides existing file.
			hid_t file = H5Fcreate(m_simulationProperties.GetOutputFileName("par").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, accesPList);
			m_fileHandle.Set(file, H5Fclose);

			H5Pclose(accesPList);

			// Create file space - a 2D matrix [edgeSize][edgeSize]
			// Create mem space  - a 2D matrix [extendedTileRows][extendedTileCols] mapped on 1D array lTempArray.
			const hsize_t rank = 2;
			const hsize_t datasetSize[2] = {hsize_t(edgeSize), hsize_t(edgeSize)};
			const hsize_t memSize[2] = {hsize_t(extendedTileRows), hsize_t(extendedTileCols)};
			
			hid_t filespace = H5Screate_simple(rank, datasetSize, nullptr);
			hid_t memspace  = H5Screate_simple(rank, memSize, nullptr);

			// Select a hyperslab to write a local submatrix into the dataset.
			const hsize_t fsStart[2] = {hsize_t((m_rank / tilesX) * tileRows), hsize_t((m_rank % tilesX) * tileCols)};
			const hsize_t count[2] = {hsize_t(tileRows), hsize_t(tileCols)};
			const hsize_t msStart[2] = {hsize_t(OFFSET), hsize_t(OFFSET)};
			H5Sselect_hyperslab(filespace, H5S_SELECT_SET, fsStart, nullptr, count, nullptr);
			H5Sselect_hyperslab(memspace, H5S_SELECT_SET, msStart, nullptr, count, nullptr);

			filespaceHandle.Set(filespace, H5Sclose);
			memspaceHandle.Set(memspace, H5Sclose);

			// Create XFER property list and set Collective IO.
			hid_t xferPList = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(xferPList, H5FD_MPIO_COLLECTIVE);
			xferPListHandle.Set(xferPList, H5Pclose);
		}
		else if(m_rank == MPI_ROOT_RANK){
			hid_t file = H5Fcreate(m_simulationProperties.GetOutputFileName("par").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
			m_fileHandle.Set(file, H5Fclose);
		}
	}
}

void ParallelHeatSolver::initMemory(){
	//// 3) Helper arrays for each process
	lTempArray1.resize(extendedTileSize);
	lTempArray2.resize(extendedTileSize);
	lDomainParams.resize(extendedTileSize);
	lDomainMap.resize(extendedTileSize);
}

void ParallelHeatSolver::initTileInfo(){
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

	// neighbour indices
	neighbourLeft = m_rank - 1;
	neighbourRight = m_rank + 1;
	neighbourTop = m_rank - tilesX;
	neighbourBottom = m_rank + tilesX;
}

void ParallelHeatSolver::initTypes(){
	// initialize types
	constexpr int ndims = 2;
	const int size[ndims] = {extendedTileRows, extendedTileCols};
	const int subsize[ndims] = {tileRows, tileCols};
	const int start[ndims] = {OFFSET, OFFSET};

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
		MPI_Type_create_resized(TYPE_ROOT_TILE_FLOAT_INITIAL, 0, tileCols * sizeof(float), &TYPE_ROOT_TILE_FLOAT);
		MPI_Type_commit(&TYPE_ROOT_TILE_FLOAT);

		MPI_Type_create_subarray(ndims, rsize, rsubsize, rstart, MPI_ORDER_C, MPI_INT, &TYPE_ROOT_TILE_INT_INITIAL);
		MPI_Type_commit(&TYPE_ROOT_TILE_INT_INITIAL);
		MPI_Type_create_resized(TYPE_ROOT_TILE_INT_INITIAL, 0, tileCols * sizeof(int), &TYPE_ROOT_TILE_INT);
		MPI_Type_commit(&TYPE_ROOT_TILE_INT);
	}
}

void ParallelHeatSolver::initRMA(){
	// init RMA windows
	if(m_simulationProperties.IsRunParallelRMA()){
		MPI_Win_create(lTempArray1.data(), extendedTileSize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &window1);
		MPI_Win_create(lTempArray2.data(), extendedTileSize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &window2);
	}
}

void ParallelHeatSolver::initMiddleColumnInfo(){
	// init for exchange of middle column temeprature
	MPI_Comm_split(MPI_COMM_WORLD, m_rank % tilesX, m_rank / tilesX, &MPI_COMM_COLS);
	middleColumnIndex = edgeSize / 2;
	middleColumnTileCol = middleColumnIndex / tileCols;
	containsMiddleColumn = (m_rank % tilesX) == middleColumnTileCol;
	middleColumnTileColIndex = middleColumnIndex % tileCols;
}

void ParallelHeatSolver::initScatterTiles(){
	//// 4) scatter tiles
	vTileCounts.resize(tilesX * tilesY);
	std::fill(vTileCounts.begin(), vTileCounts.end(), 1);

	vTileDisplacements.resize(tilesX * tilesY);
	int k = 0;
	for(int i = 0; i < tilesY; i++){
		for(int j = 0; j < tilesX; j++){
			vTileDisplacements[k++] = (i * tileRows * tilesX) + j;
		}
	}

	MPI_Scatterv(m_materialProperties.GetInitTemp().data(), vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_FLOAT, lTempArray1.data(), 1, TYPE_WORKER_TILE_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);
	MPI_Scatterv(m_materialProperties.GetInitTemp().data(), vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_FLOAT, lTempArray2.data(), 1, TYPE_WORKER_TILE_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);
	MPI_Scatterv(m_materialProperties.GetDomainParams().data(), vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_FLOAT, lDomainParams.data(), 1, TYPE_WORKER_TILE_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);
	MPI_Scatterv(m_materialProperties.GetDomainMap().data(), vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_INT, lDomainMap.data(), 1, TYPE_WORKER_TILE_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);
}

void ParallelHeatSolver::initBorderExchange(){
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
	MPI_Comm_free(&MPI_COMM_COLS);

	if(m_simulationProperties.IsRunParallelRMA()){
		MPI_Win_free(&window1);
		MPI_Win_free(&window2);
	}
}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float>>& outResult){
	float* workTempArrays[2] = {lTempArray1.data(), lTempArray2.data()};
	double startTime;

	// 5) Store start time at the root rank
	if(m_rank == MPI_ROOT_RANK){
		startTime = MPI_Wtime();
	}

	MPI_Request reqs[8];
	unsigned i = 0;

	MPI_Win* windows[2] = {&window1, &window2};

	const int lenX = tileEndX - tileStartX;
	const int lenY = tileEndY - tileStartY;
	const int rightBorderStartX = extendedTileCols - DOUBLE_OFFSET;
	const int bottomBorderStartY = extendedTileRows - DOUBLE_OFFSET;

	// open windows
	if(m_simulationProperties.IsRunParallelRMA()){
		MPI_Win_fence(MPI_MODE_NOPRECEDE, window1);
		MPI_Win_fence(MPI_MODE_NOPRECEDE, window2);
	}

	// UpdateTile(...) method can be used to evaluate heat equation over 2D tile
	//                 in parallel (using OpenMP).
	// NOTE: This method might be inefficient when used for small tiles such as 
	//       2xN or Nx2 (these might arise at edges of the tile)
	//       In this case ComputePoint may be called directly in loop.
	// 6), 7) run simulation with `m_simulationProperties.GetNumIterations()` steps
	for(size_t iter = 0; iter < m_simulationProperties.GetNumIterations(); iter++){
		// 7b) compute borders first
		if(!atLeftBorder && lenX >= OFFSET){
			UpdateTile(
				workTempArrays[0], workTempArrays[1], lDomainParams.data(), lDomainMap.data(),
				OFFSET, tileStartY, OFFSET, lenY, extendedTileCols,
				m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp()
			);
		}
		if(!atRightBorder && lenX >= OFFSET){
			UpdateTile(
				workTempArrays[0], workTempArrays[1], lDomainParams.data(), lDomainMap.data(),
				rightBorderStartX, tileStartY, OFFSET, lenY, extendedTileCols,
				m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp()
			);
		}
		if(!atTopBorder && lenY >= OFFSET){
			UpdateTile(
				workTempArrays[0], workTempArrays[1], lDomainParams.data(), lDomainMap.data(),
				tileStartX, OFFSET, lenX, OFFSET, extendedTileCols,
				m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp()
			);
		}
		if(!atBottomBorder && lenY >= OFFSET){
			UpdateTile(
				workTempArrays[0], workTempArrays[1], lDomainParams.data(), lDomainMap.data(),
				tileStartX, bottomBorderStartY, lenX, OFFSET, extendedTileCols,
				m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp()
			);
		}

		// begin exchange of computed results
		if(m_simulationProperties.IsRunParallelRMA()){
			if(!atLeftBorder){
				MPI_Put(&workTempArrays[1][leftBorderSendIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourLeft, rightBorderRecvIdx, 1, TYPE_TILE_BORDER_LR_FLOAT, *windows[1]);
			}
			if(!atRightBorder){
				MPI_Put(&workTempArrays[1][rightBorderSendIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourRight, leftBorderRecvIdx, 1, TYPE_TILE_BORDER_LR_FLOAT, *windows[1]);
			}
			if(!atTopBorder){
				MPI_Put(&workTempArrays[1][topBorderSendIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourTop, bottomBorderRecvIdx, 1, TYPE_TILE_BORDER_TB_FLOAT, *windows[1]);
			}
			if(!atBottomBorder){
				MPI_Put(&workTempArrays[1][bottomBorderSendIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourBottom, topBorderRecvIdx, 1, TYPE_TILE_BORDER_TB_FLOAT, *windows[1]);
			}
		}
		else{
			i = 0;

			if(!atLeftBorder){
				MPI_Isend(&workTempArrays[1][leftBorderSendIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourLeft, TAG_BORDER_EXCHANGE, MPI_COMM_WORLD, &reqs[i++]);
				MPI_Irecv(&workTempArrays[1][leftBorderRecvIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourLeft, TAG_BORDER_EXCHANGE, MPI_COMM_WORLD, &reqs[i++]);
			}
			if(!atRightBorder){
				MPI_Isend(&workTempArrays[1][rightBorderSendIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourRight, TAG_BORDER_EXCHANGE, MPI_COMM_WORLD, &reqs[i++]);
				MPI_Irecv(&workTempArrays[1][rightBorderRecvIdx], 1, TYPE_TILE_BORDER_LR_FLOAT, neighbourRight, TAG_BORDER_EXCHANGE, MPI_COMM_WORLD, &reqs[i++]);
			}
			if(!atTopBorder){
				MPI_Isend(&workTempArrays[1][topBorderSendIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourTop, TAG_BORDER_EXCHANGE, MPI_COMM_WORLD, &reqs[i++]);
				MPI_Irecv(&workTempArrays[1][topBorderRecvIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourTop, TAG_BORDER_EXCHANGE, MPI_COMM_WORLD, &reqs[i++]);
			}
			if(!atBottomBorder){
				MPI_Isend(&workTempArrays[1][bottomBorderSendIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourBottom, TAG_BORDER_EXCHANGE, MPI_COMM_WORLD, &reqs[i++]);
				MPI_Irecv(&workTempArrays[1][bottomBorderRecvIdx], 1, TYPE_TILE_BORDER_TB_FLOAT, neighbourBottom, TAG_BORDER_EXCHANGE, MPI_COMM_WORLD, &reqs[i++]);
			}
		}

		// compute center of the tile
		if(tileCols > DOUBLE_OFFSET && tileRows > DOUBLE_OFFSET){
			UpdateTile(
				workTempArrays[0], workTempArrays[1], lDomainParams.data(), lDomainMap.data(),
				DOUBLE_OFFSET, DOUBLE_OFFSET, tileCols - DOUBLE_OFFSET, tileRows - DOUBLE_OFFSET, extendedTileCols,
				m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp()
			);
		}

		// wait for exchange of border to finish
		if(m_simulationProperties.IsRunParallelRMA()){
			MPI_Win_fence(0, *windows[1]); // close window
			std::swap(windows[0], windows[1]);
		}
		else{
			MPI_Waitall(i, reqs, MPI_STATUSES_IGNORE);
		}

		// 7c) store data into a file 
		if(!m_simulationProperties.GetOutputFileName().empty() && ((iter % m_simulationProperties.GetDiskWriteIntensity()) == 0)){
			if(m_simulationProperties.IsUseParallelIO()){
				storeDataIntoFileParallel(iter, workTempArrays[1]);
			}
			else{
				sendMatrixToRoot(workTempArrays[1], outResult.data());

				if(m_rank == MPI_ROOT_RANK){
					StoreDataIntoFile(m_fileHandle, iter, outResult.data());
				}
			}
		}

		// ShouldPrintProgress(N) returns true if average temperature should be reported
		// by 0th process at Nth time step (using "PrintProgressReport(...)").
		// 7d) print progress (middle column temperature)
		if(ShouldPrintProgress(iter)){
			computeMiddleColAvgTemp(workTempArrays[1]);

			if(m_rank == MPI_ROOT_RANK){
				PrintProgressReport(iter, middleColAvgTemp);
			}
		}

		std::swap(workTempArrays[0], workTempArrays[1]);
	}

	// Finally "PrintFinalReport(...)" should be used to print final elapsed time and
	// average temperature in column.
	// 8) print final report
	computeMiddleColAvgTemp(workTempArrays[0]); // arrays were swapped, final values are in the first one
	if(m_rank == MPI_ROOT_RANK){
		const double elapsedTime = MPI_Wtime() - startTime;
		PrintFinalReport(elapsedTime, middleColAvgTemp, "par");
	}

	// 9) send all tiles back to root to the outResult array
	sendMatrixToRoot(workTempArrays[0], outResult.data());
}

void ParallelHeatSolver::computeMiddleColAvgTemp(const float* const data){
	if(containsMiddleColumn){
		float localTempSum = 0.0f;
		for(size_t i = OFFSET; i < tileRows + OFFSET; i++){
			localTempSum += data[(i * extendedTileCols) + OFFSET + middleColumnTileColIndex];
		}
		MPI_Reduce(&localTempSum, &middleColAvgTemp, 1, MPI_FLOAT, MPI_SUM, MPI_ROOT_RANK, MPI_COMM_COLS);
	}

	// more than one tile in X direction => root rank of middle column is not root rank => send data to root
	if(tilesX > 1){
		if(middleColumnTileCol){ // root rank of the middle column
			MPI_Send(&middleColAvgTemp, 1, MPI_FLOAT, MPI_ROOT_RANK, TAG_MIDDLE_COL, MPI_COMM_WORLD);
		}

		if(m_rank == MPI_ROOT_RANK){
			MPI_Recv(&middleColAvgTemp, 1, MPI_FLOAT, middleColumnTileCol, TAG_MIDDLE_COL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	if(m_rank == MPI_ROOT_RANK)
		middleColAvgTemp /= edgeSize;
}

void ParallelHeatSolver::sendMatrixToRoot(float* sendbuf, float* recvbuf){
	MPI_Gatherv(
		sendbuf, 1, TYPE_WORKER_TILE_FLOAT, recvbuf,
		vTileCounts.data(), vTileDisplacements.data(), TYPE_ROOT_TILE_FLOAT,
		MPI_ROOT_RANK, MPI_COMM_WORLD
	);
}

void ParallelHeatSolver::storeDataIntoFileParallel(const size_t iteration, const float* data){
	// 1. Create new HDF5 file group named as "Timestep_N", where "N" is number
	//    of current snapshot. The group is placed into root of the file "/Timestep_N".
	const std::string groupName = "Timestep_" + std::to_string(static_cast<unsigned long long>(iteration / m_simulationProperties.GetDiskWriteIntensity()));
	AutoHandle<hid_t> groupHandle(
		H5Gcreate(m_fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
		H5Gclose
	);

	{
		const hsize_t gridSize[2] = {edgeSize, edgeSize};

		// 2. Create new dataset "/Timestep_N/Temperature" which is simulation-domain
		//    sized 2D array of "float"s.
		const std::string dataSetName("Temperature");
		// 2.1 Define shape of the dataset (2D edgeSize x edgeSize array).
		AutoHandle<hid_t> dataSpaceHandle(H5Screate_simple(2, gridSize, NULL), H5Sclose);
		// 2.2 Create dataset with specified shape.
		AutoHandle<hid_t> dataSetHandle(
			H5Dcreate(
				groupHandle, dataSetName.c_str(), H5T_NATIVE_FLOAT, dataSpaceHandle,
				H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
			),
			H5Dclose
		);

		// Write the data from memory pointed by "data" into new dataset.
		H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memspaceHandle, filespaceHandle, xferPListHandle, data);
	}

	{
		// 3. Create Integer attribute in the same group "/Timestep_N/Time"
		//    in which we store number of current simulation iteration.
		const std::string attributeName("Time");

		// 3.1 Dataspace is single value/scalar.
		AutoHandle<hid_t> dataSpaceHandle(H5Screate(H5S_SCALAR), H5Sclose);

		// 3.2 Create the attribute in the group as double.
		AutoHandle<hid_t> attributeHandle(
			H5Acreate2(
				groupHandle, attributeName.c_str(), H5T_IEEE_F64LE, dataSpaceHandle,
				H5P_DEFAULT, H5P_DEFAULT
			),
			H5Aclose
		);

		// 3.3 Write value into the attribute.
		const double snapshotTime = double(iteration);
		H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
	}
}
