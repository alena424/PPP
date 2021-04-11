/**
 * @file    parallel_heat_solver.cpp
 * @author  xtesar36 <xtesar36@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2020/2021 - Project 1
 *
 * @date    2021-04-23
 */

#include "parallel_heat_solver.h"

#include <unistd.h>
#include <math.h>

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

ParallelHeatSolver::ParallelHeatSolver(SimulationProperties &simulationProps,
                                       MaterialProperties &materialProps) : BaseHeatSolver(simulationProps, materialProps),
                                                                            m_fileHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t)>(nullptr))
{
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    // std::cout << "Number of process is: " << m_size << std::endl;
    // std::cout << "I am rank: " << m_rank << std::endl;

    int matrixSize = (materialProps.GetEdgeSize() + padding * 2) * (materialProps.GetEdgeSize() + padding * 2);

    m_tempArray.resize(matrixSize);
    m_domainParams.resize(matrixSize);
    m_domainMap.resize(matrixSize);

    // 1. Open output file if its name was specified.
    AutoHandle<hid_t> myHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t)>(nullptr));

    if (!m_simulationProperties.GetOutputFileName().empty())
        myHandle.Set(H5Fcreate(simulationProps.GetOutputFileName("par").c_str(),
                               H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                     H5Fclose);

    m_simulationProperties.GetDecompGrid(globalCols, globalRows);
    InitTileVariables();
    InitRankProfile();
    InitRankOffsets();
    InitRankTypes();

    // Creating EMPTY HDF5 handle using RAII "AutoHandle" type
    //
    // AutoHandle<hid_t> myHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
    //
    // This can be particularly useful when creating handle as class member!
    // Handle CANNOT be assigned using "=" or copy-constructor, yet it can be set
    // using ".Set(/* handle */, /* close/free function */)" as in:
    // myHandle.Set(H5Fopen(...), H5Fclose);

    // Requested domain decomposition can be queried by
    // m_simulationProperties.GetDecompGrid(/* TILES IN X */, /* TILES IN Y */)
}

void ParallelHeatSolver::InitTileVariables()
{
    n = m_materialProperties.GetEdgeSize();
    tempN = n + 2 * padding;            // padding from left and right
    tileCols = n / globalCols;          // columns length = tile size X
    tileRows = n / globalRows;          // rows length = tile size Y
    blockCols = tileCols + 2 * padding; // tile cols with padding from both sides
    blockRows = tileRows + 2 * padding; // tile rows with padding from both sides
    isModeRMA = m_simulationProperties.IsRunParallelRMA();

    tile = new float[blockRows * blockCols]();    // old tile
    newTile = new float[blockRows * blockCols](); // updated tile used in computation

    // We will add borders of padding size to all parameters (it will be easier job to work with extended arrays)
    domainMap = new int[blockRows * blockCols];
    domainParams = new float[blockRows * blockCols];
}
ParallelHeatSolver::~ParallelHeatSolver()
{
}

void ParallelHeatSolver::AddPaddingToArray(float *data, int size, int padding, float *newData)
{
    int newRowSize = size + (padding * 2); // padding is from left and right
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            newData[(i + padding) * newRowSize + (j + padding)] = data[(size * i) + j];
        }
    }
}

void ParallelHeatSolver::AddPaddingToIntArray(int *data, int size, int padding, int *newData)
{
    int newRowSize = size + (padding * 2); // padding is from left and right
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            newData[(i + padding) * newRowSize + (j + padding)] = data[(size * i) + j];
        }
    }
}

/**
 * Get MPI rank within the communicator.
 * @param [in] comm - actual communicator.
 * @return rank within the comm.
 */
int ParallelHeatSolver::mpiGetCommRank(const MPI_Comm &comm)
{
    int rank = MPI_UNDEFINED;

    MPI_Comm_rank(comm, &rank);

    return rank;
} // end of mpiGetCommRank
//-----------------------------

/**
 * Get size of the communicator.
 * @param [in] comm - actual communicator.
 * @return number of ranks within the comm.
 */
int ParallelHeatSolver::mpiGetCommSize(const MPI_Comm &comm)
{
    int size = -1;
    MPI_Comm_size(comm, &size);
    return size;
} // end of mpiGetCommSize

void ParallelHeatSolver::CreateResVector(int count, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype, unsigned long size)
{
    MPI_Datatype tempType;
    MPI_Type_vector(count, 1, stride, oldtype, &tempType);
    MPI_Type_commit(&tempType);
    MPI_Type_create_resized(tempType, 0, 1 * size, newtype);
    MPI_Type_commit(newtype);
}

int *ParallelHeatSolver::GetSendCounts(int offset)
{
    int *sendCounts = new int[m_size];
    for (int i = 0; i < m_size; i++)
    {
        sendCounts[i] = offset;
    }
    return sendCounts;
}

int *ParallelHeatSolver::GetDisplacementCounts(int n)
{
    int *displacements = new int[m_size];
    displacements[0] = 0;
    int poc = 0;
    for (int i = 0; i < globalRows; i++)
    {
        for (int j = 0; j < globalCols; j++)
        {

            displacements[poc] = (j * tileCols) + (i * tileRows * n);
            poc++;
        }
    }
    return displacements;
}

void ParallelHeatSolver::InitRankOffsets()
{

    rankOffsets.startLF = 2;
    rankOffsets.endLF = blockRows - 2;
    rankOffsets.startTB = 2;
    rankOffsets.endTB = blockCols - 2;
    if (rankProfile.isBottomRank)
    {
        rankOffsets.endLF = blockRows - 4;
    }
    if (rankProfile.isTopRank)
    {
        rankOffsets.startLF = 4;
    }
    // mozna bude potreba prehodit
    if (rankProfile.isLeftRank)
    {
        rankOffsets.startTB = 4; // starting computation from fourth row
    }
    if (rankProfile.isRightRank)
    {
        rankOffsets.endTB = blockCols - 4;
    }
    if (!rankProfile.isLeftRank && !rankProfile.isRightRank && !rankProfile.isTopRank && !rankProfile.isBottomRank)
    {
        // Avoid duplicating calculation
        rankOffsets.startTB = 4;
        rankOffsets.endTB = blockCols - 4;
    }
}

void ParallelHeatSolver::InitRankProfile()
{
    rankProfile.isLeftRank = m_rank % globalCols == 0;
    rankProfile.isRightRank = m_rank % globalCols == globalCols - 1;
    rankProfile.isTopRank = m_rank < globalCols;
    rankProfile.isBottomRank = m_rank >= m_size - globalCols;
}

void ParallelHeatSolver::InitWorkingArrays()
{
    AddPaddingToArray(m_materialProperties.GetInitTemp().data(), n, padding, m_tempArray.data());
    AddPaddingToArray(m_materialProperties.GetDomainParams().data(), n, padding, m_domainParams.data());
    AddPaddingToIntArray(m_materialProperties.GetDomainMap().data(), n, padding, m_domainMap.data());
}

void ParallelHeatSolver::InitRootRankTypes()
{
    CreateResVector(blockRows, tempN, MPI_FLOAT, &MPI_COL_MAT_RES, sizeof(float)); //Create type for one column in matrix (with padding)  of type float
    CreateResVector(blockRows, tempN, MPI_INT, &MPI_COL_MAP_RES, sizeof(int));     // Create type for one column in matrix (with padding) of type int
    CreateResVector(tileRows, n, MPI_FLOAT, &MPI_COL_TILE_N_RES, sizeof(float));
}
void ParallelHeatSolver::InitRankTypes()
{
    MPI_Type_vector(blockCols, 2, blockRows, MPI_FLOAT, &MPI_COL_TILE);
    MPI_Type_commit(&MPI_COL_TILE);

    int *sizeTile = new int[2];
    sizeTile[0] = tileRows;

    int *size = new int[2];
    size[0] = blockRows;

    int *start = new int[2];
    start[0] = padding;
    // Create 1d subarray that extracts inner part of matrix (part without borders of padding size)
    MPI_Type_create_subarray(1, size, sizeTile, start, MPI_ORDER_C, MPI_FLOAT, &MPI_TILE);
    MPI_Type_commit(&MPI_TILE);

    // Create type for one row in tile
    MPI_Type_contiguous(blockRows, MPI_FLOAT, &MPI_ROW_BLOCK);
    MPI_Type_contiguous(blockRows, MPI_INT, &MPI_ROW_MAP);

    MPI_Type_commit(&MPI_ROW_BLOCK);
    MPI_Type_commit(&MPI_ROW_MAP);
}

void ParallelHeatSolver::ScatterValues(int *sendCountsTempN, int *displacementsTempN)
{

    MPI_Scatterv(m_tempArray.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, tile, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(m_tempArray.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, newTile, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(m_domainParams.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, domainParams, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(m_domainMap.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAP_RES, domainMap, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float>> &outResult)
{
    float middleColAvgTemp = 0.0f;
    std::vector<float, AlignedAllocator<float>> matrixMM(n * n); // computed final matrix
    double startTime = 0.0;

    // cout << "griiid globalCols:" << globalCols << ", globalRows:" << globalRows << " >> tileRows: " << tileRows << ", tileCols:" << tileCols << endl;
    // cout << "griiid blockCols:" << blockCols << ", blockRows:" << blockRows << " >> tempN: " << tempN << ", n:" << n << endl;
    // cout << m_rank << ": isLeftRank:" << isLeftRank << ", rankProfile.isRightRank:" << rankProfile.isRightRank << " >> rankProfile.isTopRank: " << rankProfile.isTopRank << ", rankProfile.isBottomRank:" << rankProfile.isBottomRank << endl;

    MPI_Comm MPI_COL_COMM;

    MPI_Win winNewTile;
    MPI_Win winTile;
    MPI_Win win;

    if (isModeRMA)
    {
        MPI_Win_create(newTile, blockRows * blockCols * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &winNewTile); // size of tile
        MPI_Win_create(tile, blockRows * blockCols * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &winTile);       // size of tile
    }

    // Create a col communicator using split
    MPI_Comm_split(MPI_COMM_WORLD,
                   m_rank % globalCols,
                   m_rank / globalCols,
                   &MPI_COL_COMM);

    if (m_rank == 0)
    {
        startTime = MPI_Wtime();
        InitWorkingArrays();
        InitRootRankTypes();
    }
    int *sendCountsN = GetSendCounts(tileCols);
    int *displacementsN = GetDisplacementCounts(n);

    // Allocate arrays for sendCounts and displacements
    int *sendCountsTempN = GetSendCounts(blockCols);
    int *displacementsTempN = GetDisplacementCounts(tempN);

    ScatterValues(sendCountsTempN, displacementsTempN);

    // MPI_Scatterv(m_tempArray.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, tile, blockCols, MPI_ROW_BLOCK,
    //              0, MPI_COMM_WORLD);

    // MPI_Scatterv(m_tempArray.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, newTile, blockCols, MPI_ROW_BLOCK,
    //              0, MPI_COMM_WORLD);

    // MPI_Scatterv(m_domainParams.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, domainParams, blockCols, MPI_ROW_BLOCK,
    //              0, MPI_COMM_WORLD);

    // MPI_Scatterv(m_domainMap.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAP_RES, domainMap, blockCols, MPI_ROW_BLOCK,
    //              0, MPI_COMM_WORLD);
    //printMatrix(tile, blockCols, blockRows, m_rank);
    //cout << m_rank << ": rankOffsets.startLF:" << rankOffsets.startLF << ", rankOffsets.rankOffsets.endLF:" << rankOffsets.rankOffsets.endLF << " >> rankOffsets.startTB: " << rankOffsets.startTB << ", rankOffsets.endTB:" << rankOffsets.endTB << endl;

    for (size_t iter = 0; iter < m_simulationProperties.GetNumIterations(); ++iter)
    {
        MPI_Request request[4];
        MPI_Status status[4];
        // Count only borders
        // Count left border
        if (!rankProfile.isLeftRank)
        {
            UpdateTile(tile, newTile, domainParams, domainMap, rankOffsets.startLF, 2, rankOffsets.endLF - rankOffsets.startLF, 2, blockRows,
                       m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp());
            //printMatrix(newTile, blockCols, blockRows, m_rank);
        }
        if (!rankProfile.isRightRank)
        {
            //Count right border
            UpdateTile(tile, newTile, domainParams, domainMap, rankOffsets.startLF, blockCols - 4, rankOffsets.endLF - rankOffsets.startLF, 2, blockRows,
                       m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp());
            //printMatrix(newTile, blockCols, blockRows, m_rank);
        }
        if (!rankProfile.isBottomRank)
        {
            UpdateTile(tile, newTile, domainParams, domainMap, blockRows - 4, rankOffsets.startTB, 2, rankOffsets.endTB - rankOffsets.startTB, blockRows,
                       m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp());
        }
        if (!rankProfile.isTopRank)
        {
            //Count top = right border
            UpdateTile(tile, newTile, domainParams, domainMap, 2, rankOffsets.startTB, 2, rankOffsets.endTB - rankOffsets.startTB, blockRows,
                       m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp());
        }
        if (isModeRMA)
        {
            MPI_Win_fence(0, winNewTile); // Open window
            MPI_Win_fence(0, winTile);    // Open window

            if (iter % 2 == 0)
            {
                win = winNewTile; // even iterations pointer is on winNewTile
            }
            else
            {
                win = winTile; // odd iterations pointer is on winTile
            }

            if (!rankProfile.isLeftRank)
            {
                MPI_Put(&newTile[2 * blockRows], 2, MPI_ROW_BLOCK, m_rank - 1, (blockCols - 2) * blockRows, 2, MPI_ROW_BLOCK, win); // put 2 rows to 0 index to rank on the left
            }
            if (!rankProfile.isRightRank)
            {
                MPI_Put(&newTile[(blockCols - 4) * blockRows], 2, MPI_ROW_BLOCK, m_rank + 1, 0, 2, MPI_ROW_BLOCK, win);
            }
            if (!rankProfile.isBottomRank)
            {
                MPI_Put(&newTile[blockRows - 4], 1, MPI_COL_TILE, m_rank + globalCols, 0, 1, MPI_COL_TILE, win);
            }
            if (!rankProfile.isTopRank)
            {
                MPI_Put(&newTile[2], 1, MPI_COL_TILE, m_rank - globalCols, blockRows - 2, 1, MPI_COL_TILE, win);
            }
        }
        else
        {
            // Send counted values
            if (!rankProfile.isLeftRank)
            {
                // Send left border
                MPI_Isend(&newTile[2 * blockRows], 2, MPI_ROW_BLOCK, m_rank - 1, 0, MPI_COMM_WORLD, &request[0]);
                MPI_Recv(newTile, 2, MPI_ROW_BLOCK, m_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[0] = MPI_REQUEST_NULL;
            }
            if (!rankProfile.isRightRank)
            {
                // Get right border
                MPI_Isend(&newTile[(blockCols - 4) * blockRows], 2, MPI_ROW_BLOCK, m_rank + 1, 0, MPI_COMM_WORLD, &request[1]);
                MPI_Recv(&newTile[(blockCols - 2) * blockRows], 2, MPI_ROW_BLOCK, m_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[1] = MPI_REQUEST_NULL;
            }
            if (!rankProfile.isBottomRank)
            {
                MPI_Isend(&newTile[blockRows - 4], 1, MPI_COL_TILE, m_rank + globalCols, 0, MPI_COMM_WORLD, &request[2]);
                MPI_Recv(&newTile[blockRows - 2], 1, MPI_COL_TILE, m_rank + globalCols, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[2] = MPI_REQUEST_NULL;
            }
            if (!rankProfile.isTopRank)
            {
                MPI_Isend(&newTile[2], 1, MPI_COL_TILE, m_rank - globalCols, 0, MPI_COMM_WORLD, &request[3]);
                MPI_Recv(newTile, 1, MPI_COL_TILE, m_rank - globalCols, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[3] = MPI_REQUEST_NULL;
            }
        }

        // // Count inner block
        UpdateTile(tile, newTile, domainParams, domainMap, 4, 4, blockRows - 8, blockCols - 8, blockRows,
                   m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp());
        if (isModeRMA)
        {
            MPI_Win_fence(0, winNewTile); // closing window winNewTile
            MPI_Win_fence(0, winTile);    // closing window winTile
        }
        else
        {
            MPI_Waitall(4, request, status);
        }

        ComputeMiddleColAvgTemp(
            globalCols,
            newTile,
            tileRows,
            tileCols,
            blockRows,
            &middleColAvgTemp,
            MPI_COL_COMM,
            m_rank);

        if (m_rank == 0)
        {
            PrintProgressReport(iter, middleColAvgTemp);
        }
        // Swap source and destination buffers
        std::swap(tile, newTile);
    }

    MPI_Gatherv(&tile[2 * blockRows], tileCols, MPI_TILE, matrixMM.data(), sendCountsN, displacementsN, MPI_COL_TILE_N_RES, 0, MPI_COMM_WORLD);

    mpiFlush();
    mpiPrintf(0, "---------------------------------------------------------------------\n");
    mpiFlush();
    if (m_rank == 0)
    {
        // Measure total execution time and report
        double elapsedTime = MPI_Wtime() - startTime;
        PrintFinalReport(elapsedTime, middleColAvgTemp, "par");
        printMatrix(matrixMM.data(), n, n, m_rank);
        if (m_simulationProperties.GetNumIterations() & 1)
            std::copy(matrixMM.begin(), matrixMM.end(), outResult.begin());
    }

    MPI_Type_free(&MPI_ROW_BLOCK);
    if (m_rank == 0)
    {
        MPI_Type_free(&MPI_COL_MAT_RES);
    }
    if (isModeRMA)
    {
        MPI_Win_free(&winNewTile);
        MPI_Win_free(&winTile);
    }

    //Free block of rows
    delete[] tile;
    delete[] newTile;

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
/**
 * Print block of the matrix
 * @param block     - block to print out
 * @param blockCols - number of rows in the block
 * @param nCols     - number of colls.
 */
void ParallelHeatSolver::printMatrix(float *matrix, int nRows, int nCols, int rank)
{
    std::string str;
    char val[40];

    for (int i = 0; i < nRows; i++)
    {
        str = "";
        sprintf(val, " - Rank %d = (row: %d)[", rank, i);
        str += val;

        for (int j = 0; j < nCols - 1; j++)
        {
            sprintf(val, "%.2f, ", matrix[i * nCols + j]);
            str += val;
        }
        sprintf(val, " %.2f]\n", matrix[i * nCols + nCols - 1]);
        str += val;
        printf("%s", str.c_str());
    }
} // end of printBlock

/**
 * C printf routine with selection which rank prints
 * @param who    - which rank should print. If -1 then all prints.
 * @param format - format string
 * @param ...    - other parameters
 */
void ParallelHeatSolver::mpiPrintf(int who,
                                   const char *__restrict__ format, ...)
{
    if ((who == -1) || (who == m_rank))
    {
        va_list args;
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);
    }
} // end of mpiPrintf

void ParallelHeatSolver::ComputeMiddleColAvgTemp(
    int globalCols,
    float *newTile,
    int tileRows,
    int tileCols,
    int blockRows,
    float *middleColAvgTemp,
    const MPI_Comm &comm,
    int m_rank)
{
    float localSum = 0.0f;
    float temperatureSum = 0.0f;
    bool evenColumns = globalCols % 2 == 0;

    for (int i = 2; i < tileRows + 2; i++)
    {
        if (evenColumns)
        {
            // Compute the first row
            localSum += newTile[2 * blockRows + i];
        }
        else
        {
            // Compute the middle row
            int middleRow = std::ceil(tileCols / 2.0) - 1;
            localSum += newTile[2 * blockRows + (middleRow * blockRows) + i];
        }
    }
    if (comm != MPI_COMM_NULL)
    {
        MPI_Reduce(&localSum, &temperatureSum, 1, MPI_FLOAT, MPI_SUM, 0, comm);
    }
    int middleRank = globalCols / 2;

    if (middleRank == m_rank)
    {
        //printMatrix(newTile, blockCols, blockRows, m_rank);
        *middleColAvgTemp = temperatureSum / (tileRows * mpiGetCommSize(comm));
        MPI_Send(middleColAvgTemp, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    if (m_rank == 0)
    {
        MPI_Recv(middleColAvgTemp, 1, MPI_FLOAT, middleRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/**
 * Flush stdout and call barrier to prevent message mixture.
 */
void ParallelHeatSolver::mpiFlush()
{
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    // A known hack to correctly order writes
    usleep(100);
} // end of mpiFlush