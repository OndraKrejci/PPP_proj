/**
 * @file    parallel_heat_solver.h
 * @author  xkrejc69 <xkrejc69@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2022-05-05
 */

#ifndef PARALLEL_HEAT_SOLVER_H
#define PARALLEL_HEAT_SOLVER_H

#include "base_heat_solver.h"

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 1D and 2D block grid decomposition.
 */
class ParallelHeatSolver : public BaseHeatSolver
{
	//============================================================================//
	//                            *** BEGIN: NOTE ***
	//
	// Modify this class declaration as needed.
	// This class needs to provide at least:
	// - Constructor which passes SimulationProperties and MaterialProperties
	//   to the base class. (see below)
	// - Implementation of RunSolver method. (see below)
	// 
	// It is strongly encouraged to define methods and member variables to improve 
	// readability of your code!
	//
	//                             *** END: NOTE ***
	//============================================================================//
	
public:
	/**
	 * @brief Constructor - Initializes the solver. This should include things like:
	 *        - Construct 1D or 2D grid of tiles.
	 *        - Create MPI datatypes used in the simulation.
	 *        - Open SEQUENTIAL or PARALLEL HDF5 file.
	 *        - Allocate data for local tile.
	 *        - Initialize persistent communications?
	 *
	 * @param simulationProps Parameters of simulation - passed into base class.
	 * @param materialProps   Parameters of material - passed into base class.
	 */
	ParallelHeatSolver(SimulationProperties &simulationProps, MaterialProperties &materialProps);
	virtual ~ParallelHeatSolver();

	/**
	 * @brief Run main simulation loop.
	 * @param outResult Output array which is to be filled with computed temperature values.
	 *                  The vector is pre-allocated and its size is given by dimensions
	 *                  of the input file (edgeSize*edgeSize).
	 *                  NOTE: The vector is allocated (and should be used) *ONLY*
	 *                        by master process (rank 0 in MPI_COMM_WORLD)!
	 */
	virtual void RunSolver(std::vector<float, AlignedAllocator<float> > &outResult);

protected:
	int m_rank;     ///< Process rank in global (MPI_COMM_WORLD) communicator.
	int m_size;     ///< Total number of processes in MPI_COMM_WORLD.

	// tags
	const int TAG_INIT_BORDER_TEMP = 2;
	const int TAG_INIT_BORDER_DOMAIN_PARAMS = 3;
	const int TAG_INIT_BORDER_DOMAIN_MAP = 5;
	const int TAG_BORDER_EXCHANGE = 6;
	const int TAG_MIDDLE_COL = 7;

	AutoHandle<hid_t> m_fileHandle;

	// parallel IO
	AutoHandle<hid_t> filespaceHandle;
	AutoHandle<hid_t> memspaceHandle;
	AutoHandle<hid_t> xferPListHandle;

	// matrix size
	size_t edgeSize;
	size_t matrixSize; // simulated area: edgeSize * edgeSize

	// amounts of tiles in each dimension
	int tilesX;
	int tilesY;

	// borders
	const size_t OFFSET = 2;
	const size_t DOUBLE_OFFSET = OFFSET * 2;

	// local tile data
	std::vector<float, AlignedAllocator<float>> lTempArray1;
	std::vector<float, AlignedAllocator<float>> lTempArray2;
	std::vector<float, AlignedAllocator<float>> lDomainParams;
	std::vector<int, AlignedAllocator<int>> lDomainMap;

	// tile sizes
	size_t tileCols;
	size_t tileRows;
	size_t tileSize;
	size_t extendedTileCols;
	size_t extendedTileRows;
	size_t extendedTileSize;

	// position info
	bool atLeftBorder;
	bool atRightBorder;
	bool atTopBorder;
	bool atBottomBorder;
	size_t tileStartX;
	size_t tileEndX;
	size_t tileStartY;
	size_t tileEndY;

	// counts and displacements for scatterv/gatherv operations
	std::vector<int> vTileCounts; 
	std::vector<int> vTileDisplacements; 

	// types
	MPI_Datatype TYPE_WORKER_TILE_FLOAT;
	MPI_Datatype TYPE_WORKER_TILE_INT;
	MPI_Datatype TYPE_TILE_BORDER_LR_FLOAT;
	MPI_Datatype TYPE_TILE_BORDER_TB_FLOAT;
	MPI_Datatype TYPE_TILE_BORDER_LR_INT;
	MPI_Datatype TYPE_TILE_BORDER_TB_INT;
	MPI_Datatype TYPE_ROOT_TILE_FLOAT_INITIAL;
	MPI_Datatype TYPE_ROOT_TILE_INT_INITIAL;
	MPI_Datatype TYPE_ROOT_TILE_FLOAT;
	MPI_Datatype TYPE_ROOT_TILE_INT;

	// RMA windows
	MPI_Win window1;
	MPI_Win window2;

	// tile border indices
	unsigned leftBorderSendIdx;
	unsigned rightBorderRecvIdx;
	unsigned rightBorderSendIdx;
	unsigned leftBorderRecvIdx;
	unsigned topBorderSendIdx;
	unsigned bottomBorderRecvIdx;
	unsigned bottomBorderSendIdx;
	unsigned topBorderRecvIdx;

	// neighbour indices
	int neighbourLeft;
	int neighbourRight;
	int neighbourTop;
	int neighbourBottom;

	// middle column
	MPI_Comm MPI_COMM_COLS;
	size_t middleColumnIndex;
	size_t middleColumnTileCol;
	bool containsMiddleColumn;
	size_t middleColumnTileColIndex;
	float middleColAvgTemp = 0.0f;

	/**
	 * Initializes the simulation.
	 * Runs all initialization methods in the correct order.
	 */
	void initSimulation();

	/**
	 * Saves the required decomposition and calculates tile sizes.
	 * Also check that the tile has length of at least 2 in each dimension.
	 */
	void initDecomposition();

	/**
	 * Initializes file for output.
	 * Does nothing if no output filename was specified.
	 */
	void initIO();

	/**
	 * Initializes memory to store data for each tile (resizes vectors).
	 */
	void initMemory();

	/**
	 * Computes information about the tile for every process.
	 * Computes start and end index (ignores border of the matrix and halo zones).
	 * Computes information about neighbours, whether they exist and their indexes.
	 */
	void initTileInfo();

	/**
	 * Initializes MPI datatypes.
	 */
	void initTypes();

	/**
	 * Initializes windows for RMA.
	 * Does nothing if not in RMA mode.
	 */
	void initRMA();

	/**
	 * Creates another communicator for computation of average temperature in middle column.
	 * Computes indexes of tiles containing the middle column and index of the column itself
	 */
	void initMiddleColumnInfo();

	/**
	 * Scatters the tiles (without) border to other processes from root.
	 */
	void initScatterTiles();

	/**
	 * Processes exchange halo zones of all their data arrays with their neighbours (using two-sided communication).
	 */
	void initBorderExchange();

	/**
	 * Sums the values in the middle column and send resulting value to root, which computes the average temperature.
	 */
	void computeMiddleColAvgTemp(const float* const data);

	/**
	 * Sends all tiles to root.
	 */
	void sendMatrixToRoot(float* sendbuf, float* recvbuf);

	/**
	 * Stores values in a tile for the given iteration to a file using parallel IO.
	 */
	void storeDataIntoFileParallel(const size_t iteration, const float* data);
};

#endif // PARALLEL_HEAT_SOLVER_H
