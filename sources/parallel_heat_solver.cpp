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
                                                                            m_tempArray((materialProps.GetEdgeSize() + 4) * (materialProps.GetEdgeSize() + 4)),
                                                                            m_fileHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t)>(nullptr))
{
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    std::cout << "Number of process is: " << m_size << std::endl;
    std::cout << "I am rank: " << m_rank << std::endl;

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

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float>> &outResult)
{
    int nx, ny;
    const unsigned int n = m_materialProperties.GetEdgeSize();
    const unsigned int padding = 2;
    const unsigned int tempN = n + 2 * padding; // padding from left and right
    const unsigned int tileCols = n / globalCols;
    const unsigned int tileRows = n / globalRows;
    const unsigned int blockCols = tileCols + 2 * padding;
    const unsigned int blockRows = tileRows + 2 * padding;

    float *tile = new float[blockRows * blockCols]();
    float *newTile = new float[blockRows * blockCols]();

    float *block = new float[n * blockCols](); // 4*n because we want four rows extra for borders

    float *newBlock = new float[n * blockCols]();
    int *domainMap = new int[n * blockCols];
    float *domainParams = new float[n * blockCols];
    float *matrix = nullptr; // result matrix
    std::vector<float, AlignedAllocator<float>> matrixMM(n * n);
    double startTime = 0.0;

    cout << "griiid globalCols:" << globalCols << ", globalRows:" << globalRows << " >> tileRows: " << tileRows << ", tileCols:" << tileCols << endl;
    cout << "griiid blockCols:" << blockCols << ", blockRows:" << blockRows << " >> tempN: " << tempN << ", n:" << n << endl;

    m_simulationProperties.GetDecompGrid(nx, ny);
    // Declare datatype for all columbs in Matrix
    MPI_Datatype MPI_COL_MAT;
    MPI_Datatype MPI_COL_MAT_RES;

    MPI_Datatype MPI_COL_TILE;
    MPI_Datatype MPI_COL_TILE_RES;

    MPI_Datatype MPI_ROW_TILE;
    MPI_Datatype MPI_ROW_TILE_RES;

    MPI_Datatype MPI_ROW_BLOCK;
    MPI_Datatype MPI_ROW_MAP;

    MPI_Datatype MPI_COL_MAP;
    MPI_Datatype MPI_COL_MAP_RES;

    MPI_Datatype MPI_GLOBAL_TILE;
    MPI_Datatype MPI_GLOBAL_TILE_RES;
    MPI_Datatype MPI_TILE;
    MPI_Datatype MPI_TILE_RES;

    if (m_rank == 0)
    {
        startTime = MPI_Wtime();
        matrix = new float[n * n];

        AddPaddingToArray(m_materialProperties.GetInitTemp().data(), n, padding, m_tempArray.data());
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
        MPI_Type_vector(blockRows, 1, tempN, MPI_FLOAT, &MPI_COL_MAT);
        MPI_Type_commit(&MPI_COL_MAT);
        MPI_Type_create_resized(MPI_COL_MAT, 0, 1 * sizeof(float), &MPI_COL_MAT_RES);
        MPI_Type_commit(&MPI_COL_MAT_RES);

        MPI_Type_vector(tileRows, 1, n, MPI_FLOAT, &MPI_COL_TILE);
        MPI_Type_commit(&MPI_COL_TILE);
        MPI_Type_create_resized(MPI_COL_TILE, 0, 1 * sizeof(float), &MPI_COL_TILE_RES);
        MPI_Type_commit(&MPI_COL_TILE_RES);
    }
    // MPI_Type_vector(blockRows, blockRows, padding + blockCols, MPI_FLOAT, &MPI_ROW_TILE);
    // MPI_Type_commit(&MPI_COL_TILE);
    // MPI_Type_create_resized(MPI_COL_TILE, 0, 1 * sizeof(float), &MPI_COL_TILE_RES);
    // MPI_Type_commit(&MPI_COL_TILE_RES);

    MPI_Type_vector(n, 1, n, MPI_INT, &MPI_COL_MAP);
    MPI_Type_commit(&MPI_COL_MAP);
    MPI_Type_create_resized(MPI_COL_MAP, 0, 1 * sizeof(float), &MPI_COL_MAP_RES);
    MPI_Type_commit(&MPI_COL_MAP_RES);

    mpiFlush();
    mpiPrintf(0, "---------------------------------------------------------------------\n");
    mpiFlush();

    int *sizeTile = new int[2];
    sizeTile[0] = tileRows;

    int *size = new int[2];
    size[0] = blockRows;

    int *start = new int[2];
    start[0] = padding;
    MPI_Type_create_subarray(1, size, sizeTile, start, MPI_ORDER_C, MPI_FLOAT, &MPI_TILE);
    MPI_Type_commit(&MPI_TILE);

    MPI_Type_contiguous(blockRows, MPI_FLOAT, &MPI_ROW_BLOCK);

    MPI_Type_contiguous(n * blockCols, MPI_INT, &MPI_ROW_MAP);

    MPI_Type_commit(&MPI_ROW_BLOCK);
    MPI_Type_commit(&MPI_ROW_MAP);

    // Allocate arrays for sendCounts and displacements
    int *sendCounts = new int[m_size];
    int *displacements = new int[m_size];

    displacements[0] = 0;
    for (int i = 0; i < m_size; i++)
    {
        // if (i == 0 || i == m_size - 1)
        // {
        //     // Send 2 blocks less
        //     sendCounts[i] = blockCols - 2;
        // }
        // else
        // {
        sendCounts[i] = blockCols;
        //}
    }
    // for (int i = 1; i < m_size; i++)
    // {
    //     displacements[i] = i * tileCols - 2;
    // }
    for (int i = 0; i < globalRows; i++)
    {
        for (int j = 0; j < globalCols; j++)
        {

            displacements[i * globalRows + j] = (j * tileCols) + (i * tileRows * tempN);

            cout << m_rank << ": " << displacements[i * globalRows + j] << endl;
        }
    }

    int receivedPosition = 0;
    if (m_rank == 0)
    {
        // First rank starts in 3rd block
        receivedPosition = 2 * n;
    }

    MPI_Scatterv(m_tempArray.data(), sendCounts, displacements, MPI_COL_MAT_RES, tile, blockCols, MPI_ROW_BLOCK,
                 0, MPI_COMM_WORLD);

    // MPI_Scatterv(m_materialProperties.GetInitTemp().data(), sendCounts, displacements, MPI_COL_MAT_RES, &newBlock[receivedPosition], blockCols, MPI_ROW_BLOCK,
    //              0, MPI_COMM_WORLD);

    // MPI_Scatterv(m_materialProperties.GetDomainParams().data(), sendCounts, displacements, MPI_COL_MAT_RES, &domainParams[receivedPosition], blockCols, MPI_ROW_BLOCK,
    //              0, MPI_COMM_WORLD);

    // MPI_Scatterv(m_materialProperties.GetDomainMap().data(), sendCounts, displacements, MPI_COL_MAP_RES, &domainMap[receivedPosition], blockCols, MPI_ROW_BLOCK,
    //              0, MPI_COMM_WORLD);

    printMatrix(tile, blockCols, blockRows, m_rank);

    mpiFlush();
    mpiPrintf(0, "----------------------------END-----------------------------------------\n");
    mpiFlush();

    float middleColAvgTemp = 0.0f;
    // Begin iterative simulation main loop
    bool debug = false;
    if (debug)
    {
        for (size_t iter = 0; iter < m_simulationProperties.GetNumIterations(); ++iter)
        {
            MPI_Request request[2];
            MPI_Status status[2];
            // Count only borders
            // Count left border
            if (m_rank != 0)
            {
                for (unsigned int i = 2; i < 4; ++i)
                {
                    for (unsigned int j = 2; j < n - 2; ++j)
                    {
                        ComputePoint(block, newBlock,
                                     domainParams,
                                     domainMap,
                                     i, j,
                                     n,
                                     m_simulationProperties.GetAirFlowRate(),
                                     m_materialProperties.GetCoolerTemp());
                    }
                }
            }
            if (m_rank != m_size - 1)
            {
                // Count right border
                for (unsigned int i = blockCols - 4; i < blockCols - 2; ++i)
                {
                    for (unsigned int j = 2; j < n - 2; ++j)
                    {
                        ComputePoint(block, newBlock,
                                     domainParams,
                                     domainMap,
                                     i, j,
                                     n,
                                     m_simulationProperties.GetAirFlowRate(),
                                     m_materialProperties.GetCoolerTemp());
                    }
                }
            }
            // Send counted values
            if (m_rank != 0)
            {
                // Send left border
                MPI_Isend(&block[2 * n], 2, MPI_ROW_BLOCK, m_rank - 1, 0, MPI_COMM_WORLD, &request[0]);
                MPI_Recv(block, 2, MPI_ROW_BLOCK, m_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[0] = MPI_REQUEST_NULL;
            }
            if (m_rank != m_size - 1)
            {
                // Get right border
                MPI_Isend(&block[(blockCols - 4) * n], 2, MPI_ROW_BLOCK, m_rank + 1, 0, MPI_COMM_WORLD, &request[1]);
                MPI_Recv(&block[(blockCols - 2) * n], 2, MPI_ROW_BLOCK, m_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                request[1] = MPI_REQUEST_NULL;
            }
            // Count inner block
            for (unsigned int i = 4; i < blockCols - 4; ++i)
            {
                for (unsigned int j = 2; j < n - 2; ++j)
                {
                    ComputePoint(block, newBlock,
                                 domainParams,
                                 domainMap,
                                 i, j,
                                 n,
                                 m_simulationProperties.GetAirFlowRate(),
                                 m_materialProperties.GetCoolerTemp());
                }
            }

            MPI_Waitall(2, request, status);
            int middleRank = n / (2 * tileCols);
            if (m_rank == middleRank)
            {
                // Use first row
                middleColAvgTemp = ComputeMiddleColAvgTemp(&newBlock[2 * n]);
                MPI_Send(&middleColAvgTemp, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            }
            // Swap source and destination buffers

            if (m_rank == 0)
            {
                MPI_Recv(&middleColAvgTemp, 1, MPI_FLOAT, middleRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                PrintProgressReport(iter, middleColAvgTemp);
            }
            std::swap(block, newBlock);
            printMatrix(block, blockCols, n, m_rank);
        }
    }
    int *counts = new int[m_size];
    //int disps = new int[m_size];
    int displ[3] = {0, 8};
    for (int i = 0; i < m_size; i++)
    {
        //disps[i] = 1;
        counts[i] = tileCols;
        //cout << "count " << i << ": " << counts[i] << endl;
    }

    //MPI_Gatherv(tile, tileCols, MPI_TILE, matrixMM.data(), counts, disps, MPI_GLOBAL_TILE_RES, 0, MPI_COMM_WORLD);

    MPI_Gather(&tile[2 * blockRows], tileCols, MPI_TILE, matrixMM.data(), tileCols, MPI_COL_TILE_RES, 0, MPI_COMM_WORLD);

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
    for (size_t i = 0; i < m_materialProperties.GetEdgeSize(); ++i)
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