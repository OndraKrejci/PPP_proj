/**
 * @file    parallel_heat_solver.cpp
 * @author  xkrejc69 <xkrejc69@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2020/2021 - Project 1
 *
 * @date    2021-04-DD
 */

#include "parallel_heat_solver.h"

#include <hdf5.h>

using namespace std;

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
    size_t tileCols = edgeSize / tilesX;
    size_t tileRows = edgeSize / tilesY;
    size_t extendedTileCols = tileCols + 4;
    size_t extendedTileRows = tileRows + 4;
    size_t extendedTileSize = extendedTileCols * extendedTileRows;

    //// 3) Helper arrays for each process

    lTempArray1.resize(extendedTileSize);
    lTempArray2.resize(extendedTileSize);
    lDomainParams.resize(extendedTileSize);
    lDomainMap.resize(extendedTileSize);

    // additional border removes the need to handle the borders as a special case
    if(m_rank == MPI_ROOT_RANK){
        const size_t extendedEdge = edgeSize + 4;
        const size_t extendedArrayLen = extendedEdge * extendedEdge;
        rootTempArray.resize(extendedArrayLen, 0.0);
        rootDomainParams.resize(extendedArrayLen, 0.0);
        rootDomainMap.resize(extendedArrayLen, 0);
    }
}

ParallelHeatSolver::~ParallelHeatSolver(){
}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float> > &outResult){
    // UpdateTile(...) method can be used to evaluate heat equation over 2D tile
    //                 in parallel (using OpenMP).
    // NOTE: This method might be inefficient when used for small tiles such as 
    //       2xN or Nx2 (these might arise at edges of the tile)
    //       In this case ComputePoint may be called directly in loop.
    
    // ShouldPrintProgress(N) returns true if average temperature should be reported
    // by 0th process at Nth time step (using "PrintProgressReport(...)").
    
    // Finally "PrintFinalReport(...)" should be used to print final elapsed time and
    // average temperature in column.
}
