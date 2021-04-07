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

    std::cout << "Number of process is: " << m_size << std::endl;
    std::cout << "I am rank: " << m_rank << std::endl;

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

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float>> &outResult)
{
    int nx, ny;
    const unsigned int n = m_materialProperties.GetEdgeSize();
    const unsigned int tempN = n + 2 * padding; // padding from left and right
    const unsigned int tileCols = n / globalCols;
    const unsigned int tileRows = n / globalRows;
    const unsigned int blockCols = tileCols + 2 * padding;
    const unsigned int blockRows = tileRows + 2 * padding;

    float *tile = new float[blockRows * blockCols]();
    float *newTile = new float[blockRows * blockCols]();

    float *block = new float[n * blockCols]();    // 4*n because we want four rows extra for borders
    float *newBlock = new float[n * blockCols](); // depraceted

    int *domainMap = new int[blockRows * blockCols];
    float *domainParams = new float[blockRows * blockCols];
    float *matrix = nullptr; // result matrix
    std::vector<float, AlignedAllocator<float>> matrixMM(n * n);
    double startTime = 0.0;

    bool isLeftRank = m_rank % globalCols == 0;
    bool isRightRank = m_rank % globalCols == globalCols - 1;
    bool isTopRank = m_rank < globalCols;
    bool isBottomRank = m_rank >= m_size - globalCols;

    cout << "griiid globalCols:" << globalCols << ", globalRows:" << globalRows << " >> tileRows: " << tileRows << ", tileCols:" << tileCols << endl;
    cout << "griiid blockCols:" << blockCols << ", blockRows:" << blockRows << " >> tempN: " << tempN << ", n:" << n << endl;
    cout << m_rank << ": isLeftRank:" << isLeftRank << ", isRightRank:" << isRightRank << " >> isTopRank: " << isTopRank << ", isBottomRank:" << isBottomRank << endl;

    // m_simulationProperties.GetDecompGrid(nx, ny);
    // Declare datatype for all columbs in Matrix
    MPI_Datatype MPI_COL_MAT;
    MPI_Datatype MPI_COL_MAT_RES;

    MPI_Datatype MPI_COL_TILE_N;
    MPI_Datatype MPI_COL_TILE_N_RES;

    MPI_Datatype MPI_ROW_TILE;
    MPI_Datatype MPI_ROW_TILE_RES;

    MPI_Datatype MPI_ROW_BLOCK;
    MPI_Datatype MPI_ROW_MAP;

    MPI_Datatype MPI_COL_MAP;

    MPI_Datatype MPI_COL_TILE;
    MPI_Datatype MPI_COL_MAP_RES;

    MPI_Datatype MPI_GLOBAL_TILE;
    MPI_Datatype MPI_GLOBAL_TILE_RES;
    MPI_Datatype MPI_TILE;
    MPI_Datatype MPI_TILE_RES;

    MPI_Comm MPI_COL_COMM;
    //Create a col communicator using split.
    MPI_Comm_split(MPI_COMM_WORLD,
                   m_rank % globalCols,
                   m_rank / globalCols,
                   &MPI_COL_COMM);
    //bool isMiddleRank = mpiGetCommRank(MPI_COL_COMM) == globalCols / 2;

    // if (mpiGetCommRank(MPI_COL_COMM) == globalCols / 2)
    // {
    // Middle rank
    cout << m_rank << "(" << mpiGetCommRank(MPI_COL_COMM) << "/" << mpiGetCommSize(MPI_COL_COMM) << ") " << m_rank % globalCols
         << ": I am middle rank" << endl;
    // }
    // else
    // {
    //     cout << m_rank << "(" << mpiGetCommRank(MPI_COL_COMM) << ") " << m_rank % globalCols
    //          << ": I am not middle rank" << endl;
    // }

    if (m_rank == 0)
    {
        startTime = MPI_Wtime();
        matrix = new float[n * n];

        AddPaddingToArray(m_materialProperties.GetInitTemp().data(), n, padding, m_tempArray.data());
        AddPaddingToArray(m_materialProperties.GetDomainParams().data(), n, padding, m_domainParams.data());
        AddPaddingToIntArray(m_materialProperties.GetDomainMap().data(), n, padding, m_domainMap.data());

        // Create subarray
        int *sizeTile = new int[2];
        sizeTile[0] = tileRows;
        sizeTile[1] = tileCols;

        int *size = new int[2];
        size[0] = n;
        size[1] = n;

        int *start = new int[2];
        start[0] = 0;
        start[1] = 0;
        MPI_Type_create_subarray(2, size, sizeTile, start, MPI_ORDER_C, MPI_FLOAT, &MPI_GLOBAL_TILE);
        MPI_Type_commit(&MPI_GLOBAL_TILE);
        MPI_Type_create_resized(MPI_GLOBAL_TILE, 0, 1 * sizeof(float), &MPI_GLOBAL_TILE_RES);
        MPI_Type_commit(&MPI_GLOBAL_TILE_RES);

        printMatrix(m_tempArray.data(), tempN, tempN, m_rank);
        // printMatrix(m_materialProperties.GetDomainParams().data(), n, n, m_rank);
        MPI_Type_vector(blockRows, 1, tempN, MPI_FLOAT, &MPI_COL_MAT);
        MPI_Type_commit(&MPI_COL_MAT);
        MPI_Type_create_resized(MPI_COL_MAT, 0, 1 * sizeof(float), &MPI_COL_MAT_RES);
        MPI_Type_commit(&MPI_COL_MAT_RES);

        MPI_Type_vector(blockRows, 1, tempN, MPI_INT, &MPI_COL_MAP);
        MPI_Type_commit(&MPI_COL_MAP);
        MPI_Type_create_resized(MPI_COL_MAP, 0, 1 * sizeof(int), &MPI_COL_MAP_RES);
        MPI_Type_commit(&MPI_COL_MAP_RES);

        MPI_Type_vector(tileRows, 1, n, MPI_FLOAT, &MPI_COL_TILE_N);
        MPI_Type_commit(&MPI_COL_TILE_N);
        MPI_Type_create_resized(MPI_COL_TILE_N, 0, 1 * sizeof(float), &MPI_COL_TILE_N_RES);
        MPI_Type_commit(&MPI_COL_TILE_N_RES);
    }
    // MPI_Type_vector(blockRows, blockRows, padding + blockCols, MPI_FLOAT, &MPI_ROW_TILE);
    // MPI_Type_commit(&MPI_COL_TILE_N);
    // MPI_Type_create_resized(MPI_COL_TILE_N, 0, 1 * sizeof(float), &MPI_COL_TILE_N_RES);
    // MPI_Type_commit(&MPI_COL_TILE_N_RES);

    mpiFlush();
    mpiPrintf(0, "---------------------------------------------------------------------\n");
    mpiFlush();

    MPI_Type_vector(blockCols, 2, blockRows, MPI_FLOAT, &MPI_COL_TILE);
    MPI_Type_commit(&MPI_COL_TILE);
    // MPI_Type_create_resized(MPI_COL_MAT, 0, 1 * sizeof(float), &MPI_COL_TILE_N);
    // MPI_Type_commit(&MPI_COL_MAT_RES);

    int *sizeTile = new int[2];
    sizeTile[0] = tileRows;

    int *size = new int[2];
    size[0] = blockRows;

    int *start = new int[2];
    start[0] = padding;
    MPI_Type_create_subarray(1, size, sizeTile, start, MPI_ORDER_C, MPI_FLOAT, &MPI_TILE);
    MPI_Type_commit(&MPI_TILE);

    MPI_Type_contiguous(blockRows, MPI_FLOAT, &MPI_ROW_BLOCK);
    MPI_Type_contiguous(blockRows, MPI_INT, &MPI_ROW_MAP);

    MPI_Type_commit(&MPI_ROW_BLOCK);
    MPI_Type_commit(&MPI_ROW_MAP);

    // Allocate arrays for sendCounts and displacements
    int *sendCountsTempN = new int[m_size];
    int *sendCountsN = new int[m_size];

    int *sendCounts = new int[m_size];
    int *displacementsTempN = new int[m_size];
    int *displacementsN = new int[m_size];

    displacementsTempN[0] = 0;
    displacementsN[0] = 0;
    for (int i = 0; i < m_size; i++)
    {
        sendCountsTempN[i] = blockCols;
        sendCountsN[i] = tileCols;
    }
    int poc = 0;
    for (int i = 0; i < globalRows; i++)
    {
        for (int j = 0; j < globalCols; j++)
        {

            displacementsTempN[poc] = (j * tileCols) + (i * tileRows * tempN);
            displacementsN[poc] = (j * tileCols) + (i * tileRows * n);
            poc++;
        }
    }

    MPI_Scatterv(m_tempArray.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, tile, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(m_tempArray.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, newTile, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(m_domainParams.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAT_RES, domainParams, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(m_domainMap.data(), sendCountsTempN, displacementsTempN, MPI_COL_MAP_RES, domainMap, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);

    printMatrix(tile, blockCols, blockRows, m_rank);

    mpiFlush();
    mpiPrintf(0, "----------------------------END-----------------------------------------\n");
    mpiFlush();

    float middleColAvgTemp = 0.0f;
    // Begin iterative simulation main loop
    bool debug = true;
    int startLF = 2;
    int endLF = blockRows - 2;
    int startTB = 2;
    int endTB = blockCols - 2;
    if (isBottomRank)
    {
        startLF = 4;
    }
    if (isTopRank)
    {
        endLF = blockRows - 4;
    }
    // mozna bude potreba prehodit
    if (isLeftRank)
    {
        startTB = 4; // starting computation from fourth row
    }
    if (isRightRank)
    {
        endTB = blockCols - 4;
    }
    cout << m_rank << ": startLF:" << startLF << ", endLF:" << endLF << " >> startTB: " << startTB << ", endTB:" << endTB << endl;

    if (debug)
    {
        for (size_t iter = 0; iter < m_simulationProperties.GetNumIterations(); ++iter)
        {
            MPI_Request request[4];
            MPI_Status status[4];
            // Count only borders
            // Count left border
            if (!isLeftRank)
            {
                for (unsigned int i = 2; i < 4; ++i) // Start from second row
                {
                    for (unsigned int j = startLF; j < endLF; ++j) //
                    {
                        ComputePoint(tile, newTile,
                                     domainParams,
                                     domainMap,
                                     i, j,
                                     blockRows,
                                     m_simulationProperties.GetAirFlowRate(),
                                     m_materialProperties.GetCoolerTemp());
                    }
                }
                //printMatrix(newTile, blockCols, blockRows, m_rank);
            }
            if (!isRightRank)
            {
                //Count right border
                for (unsigned int i = blockCols - 4; i < blockCols - 2; ++i)
                {
                    for (unsigned int j = startLF; j < endLF; ++j)
                    {
                        ComputePoint(tile, newTile,
                                     domainParams,
                                     domainMap,
                                     i, j,
                                     blockRows,
                                     m_simulationProperties.GetAirFlowRate(),
                                     m_materialProperties.GetCoolerTemp());
                    }
                }

                //printMatrix(newTile, blockCols, blockRows, m_rank);
            }
            if (!isBottomRank)
            {
                // for (unsigned int j = 2 + 2; j < blockRows - 2 - 2; ++j)
                //for (unsigned int i = startTB; i < endTB; ++i) // We need to go throught all the rows
                for (unsigned int i = startTB; i < endTB; ++i)
                {
                    //for (unsigned int j = 2 + 2; j < 4 + 2; ++j) //Columbs
                    for (unsigned int j = blockRows - 4; j < blockRows - 2; ++j) // We need to go throught all the rows
                    {
                        ComputePoint(tile, newTile,
                                     domainParams,
                                     domainMap,
                                     i, j,
                                     blockRows,
                                     m_simulationProperties.GetAirFlowRate(),
                                     m_materialProperties.GetCoolerTemp());
                    }
                }
                cout << m_rank << " computing" << endl;
                printMatrix(newTile, blockCols, blockRows, m_rank);
                //printMatrix(newTile, blockCols, blockRows, m_rank);
            }
            if (!isTopRank)
            {
                //Count top = right border
                //for (unsigned int i = startTB; i < endTB; ++i)
                for (unsigned int i = startTB; i < endTB; ++i)
                {
                    //for (unsigned int j = blockCols - 4 - 2; j < blockCols - 2 - 2; ++j)
                    for (unsigned int j = 2; j < 4; ++j)
                    {
                        ComputePoint(tile, newTile,
                                     domainParams,
                                     domainMap,
                                     i, j,
                                     blockRows,
                                     m_simulationProperties.GetAirFlowRate(),
                                     m_materialProperties.GetCoolerTemp());
                    }
                }
            }
            // Send counted values
            if (!isLeftRank)
            {
                // Send left border
                MPI_Isend(&newTile[2 * blockRows], 2, MPI_ROW_BLOCK, m_rank - 1, 0, MPI_COMM_WORLD, &request[0]);
                MPI_Recv(newTile, 2, MPI_ROW_BLOCK, m_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[0] = MPI_REQUEST_NULL;
            }
            if (!isRightRank)
            {
                // Get right border
                //cout << m_rank << " Sending: " << m_rank + 1 << endl;
                //printMatrix(&newTile[(blockCols - 4) * blockRows], 4, blockRows, m_rank);
                MPI_Isend(&newTile[(blockCols - 4) * blockRows], 2, MPI_ROW_BLOCK, m_rank + 1, 0, MPI_COMM_WORLD, &request[1]);
                MPI_Recv(&newTile[(blockCols - 2) * blockRows], 2, MPI_ROW_BLOCK, m_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[1] = MPI_REQUEST_NULL;
            }
            if (!isBottomRank)
            {
                //cout << m_rank << " Sending: " << m_rank + globalCols << endl;
                // Send/Receive left column (bottom tile)

                MPI_Isend(&newTile[blockRows - 4], 1, MPI_COL_TILE, m_rank + globalCols, 0, MPI_COMM_WORLD, &request[2]);
                MPI_Recv(&newTile[blockRows - 2], 1, MPI_COL_TILE, m_rank + globalCols, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[2] = MPI_REQUEST_NULL;
            }
            if (!isTopRank)
            {
                //int topRankReceiver = m_rank - globalCols;
                //cout << m_rank << ": sending ..(" << iter << ") " << endl;
                //printMatrix(&newTile[(blockRows - 4)], 1, 4, m_rank);

                MPI_Isend(&newTile[2], 1, MPI_COL_TILE, m_rank - globalCols, 0, MPI_COMM_WORLD, &request[3]);
                MPI_Recv(newTile, 1, MPI_COL_TILE, m_rank - globalCols, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[3] = MPI_REQUEST_NULL;
            }

            // // Count inner block
            for (unsigned int i = 4; i < blockCols - 4; ++i)
            {
                for (unsigned int j = 2 + 2; j < blockRows - 4; ++j)
                {
                    //cout << m_rank << ": counting " << i << ", " << j << endl;
                    ComputePoint(tile, newTile,
                                 domainParams,
                                 domainMap,
                                 i, j,
                                 blockRows,
                                 m_simulationProperties.GetAirFlowRate(),
                                 m_materialProperties.GetCoolerTemp());
                }
            }

            MPI_Waitall(4, request, status);
            if (m_rank == 0)
            {
                printMatrix(newTile, blockCols, blockRows, m_rank);
            }

            //int middleRank = n / (2 * tileCols);
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

            if (MPI_COL_COMM != MPI_COMM_NULL)
            {
                MPI_Reduce(&localSum, &temperatureSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COL_COMM);
            }
            int middleRank = globalCols / 2;
            if (middleRank == m_rank)
            {
                //printMatrix(newTile, blockCols, blockRows, m_rank);
                middleColAvgTemp = temperatureSum / (tileRows * mpiGetCommSize(MPI_COL_COMM));
                cout << m_rank << ": middle temparature is: " << temperatureSum << ", " << middleColAvgTemp << endl;
                MPI_Send(&middleColAvgTemp, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            }

            // // // Swap source and destination buffers
            std::swap(tile, newTile);

            if (m_rank == 0)
            {
                MPI_Recv(&middleColAvgTemp, 1, MPI_FLOAT, middleRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                PrintProgressReport(iter, middleColAvgTemp);
            }
            // // debug
            // //printMatrix(tile, blockCols, blockRows, m_rank);
            MPI_Gatherv(&tile[2 * blockRows], tileCols, MPI_TILE, matrixMM.data(), sendCountsN, displacementsN, MPI_COL_TILE_N_RES, 0, MPI_COMM_WORLD);
            mpiFlush();
            mpiPrintf(0, "---------------------------------------------------------------------\n");
            mpiFlush();
            if (m_rank == 0)
            {
                // printMatrix(matrixMM.data(), n, n, m_rank);
            }
        }
    }

    MPI_Gatherv(&tile[2 * blockRows], tileCols, MPI_TILE, matrixMM.data(), sendCountsN, displacementsN, MPI_COL_TILE_N_RES, 0, MPI_COMM_WORLD);

    mpiFlush();
    mpiPrintf(0, "---------------------------------------------------------------------\n");
    mpiFlush();
    if (m_rank == 0)
    {
        // Measure total execution time and report
        // double elapsedTime = MPI_Wtime() - startTime;
        // PrintFinalReport(elapsedTime, middleColAvgTemp, "par");
        printMatrix(matrixMM.data(), n, n, m_rank);
        // if (m_simulationProperties.GetNumIterations() & 1)
        //     std::copy(matrixMM.begin(), matrixMM.end(), outResult.begin());
    }

    MPI_Type_free(&MPI_ROW_BLOCK);
    if (m_rank == 0)
    {
        MPI_Type_free(&MPI_COL_MAT_RES);
        MPI_Type_free(&MPI_COL_MAT);
        //delete [] matrix;
    }

    //Free block of rows
    delete[] block;
    delete[] newBlock;

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

float ParallelHeatSolver::ComputeMiddleColAvgTemp(const float *data) const
{
    float middleColAvgTemp = 0.0f;
    for (size_t i = 2; i < m_materialProperties.GetEdgeSize() + 2; ++i)
        middleColAvgTemp += data[i];
    return middleColAvgTemp / float(m_materialProperties.GetEdgeSize());
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