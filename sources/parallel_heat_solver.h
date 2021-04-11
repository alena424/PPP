/**
 * @file    parallel_heat_solver.h
 * @author  xlogin00 <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2020/2021 - Project 1
 *
 * @date    2021-MM-DD
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
    virtual void RunSolver(std::vector<float, AlignedAllocator<float>> &outResult);

    void printMatrix(float *matrix, int nRows, int nCols, int rank);

    void mpiPrintf(int who, const char *__restrict__ format, ...);
    void mpiFlush();

protected:
    int m_rank;                     ///< Process rank in global (MPI_COMM_WORLD) communicator.
    int m_size;                     ///< Total number of processes in MPI_COMM_WORLD.
    int globalCols;                 ///< Domain decomposition in X axis.
    int globalRows;                 ///< Domain decomposition in Y axis.
    const unsigned int padding = 2; //< Size of surrounding that we need to count

    unsigned int n;         // m_materialProperties.GetEdgeSize()
    unsigned int tempN;     // m_materialProperties.GetEdgeSize() with padding from left and right
    unsigned int tileCols;  // tile cols size without padding, columns length = tile size X
    unsigned int tileRows;  // tile rows size without padding, rows length = tile size Y
    unsigned int blockCols; // tile cols with padding from both sides
    unsigned int blockRows; // tile rows with padding from both sides

    float *tile;
    float *newTile;
    int *domainMap;
    float *domainParams;

    bool isModeRMA; // = m_simulationProperties.IsRunParallelRMA()

    std::vector<float, AlignedAllocator<float>> m_tempArray;
    std::vector<float, AlignedAllocator<float>> m_domainParams;
    std::vector<int, AlignedAllocator<int>> m_domainMap;
    struct RankProfile
    {
        bool isLeftRank;   // is the rank on the left edge
        bool isRightRank;  // is the rank on the right edge
        bool isTopRank;    // is the rank on the top edge
        bool isBottomRank; // is the rank on the bottom edge
    };

    struct RankOffsets
    {
        int startLF;
        int endLF;
        int startTB;
        int endTB;
    };

    RankProfile rankProfile;
    RankOffsets rankOffsets;

    // Root rank types
    MPI_Datatype MPI_COL_MAT_RES;
    MPI_Datatype MPI_COL_MAP_RES;
    MPI_Datatype MPI_COL_TILE_N_RES;

    // All rank types
    MPI_Datatype MPI_ROW_BLOCK;
    MPI_Datatype MPI_ROW_MAP;
    MPI_Datatype MPI_TILE;
    MPI_Datatype MPI_COL_TILE;

    AutoHandle<hid_t>
        m_fileHandle;
    void AddPaddingToArray(float *data, int size, int padding, float *newData);
    void AddPaddingToIntArray(int *data, int size, int padding, int *newData);
    int mpiGetCommRank(const MPI_Comm &comm);
    int mpiGetCommSize(const MPI_Comm &comm);
    void CreateResVector(int count, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype, unsigned long size);
    int *GetSendCounts(int offset);
    int *GetDisplacementCounts(int n);
    void ComputeMiddleColAvgTemp(
        int globalCols,
        float *newTile,
        int tileRows,
        int tileCols,
        int blockRows,
        float *middleColAvgTemp,
        const MPI_Comm &comm,
        int rank);
    void InitRankProfile();
    void InitTileVariables();
    void InitRankOffsets();
    void InitWorkingArrays();
    void InitRootRankTypes();
    void InitRankTypes();
    void ScatterValues(int *sendCountsTempN, int *displacementsTempN);
};

#endif // PARALLEL_HEAT_SOLVER_H
