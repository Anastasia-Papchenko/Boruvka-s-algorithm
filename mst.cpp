#include "defs.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <mpi.h>
#include <cassert>
#include <cstdlib>  
#include <utility>  

using namespace std;

int rank_, size_, lgsize;
uint32_t TotVertices;

static int local_n;
static vertex_id_t*    local2global    = nullptr;
static edge_id_t*      localEdgeIdArr  = nullptr;

struct Edge {
    vertex_id_t startV;
    vertex_id_t endV;
    weight_t    weight;
    edge_id_t   edge_id;
};

MPI_Datatype MPI_Edge;

void create_mpi_edge_type() {
    Edge dummy = {};
    int          block_lengths[4] = {1, 1, 1, 1};
    MPI_Aint     displacements[4];
    MPI_Datatype types[4] = {
        MPI_UINT32_T,  
        MPI_UINT32_T,   
        MPI_DOUBLE,     
        MPI_UINT64_T    
    };

    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.startV,  &displacements[0]);
    MPI_Get_address(&dummy.endV,    &displacements[1]);
    MPI_Get_address(&dummy.weight,  &displacements[2]);
    MPI_Get_address(&dummy.edge_id, &displacements[3]);

    for (int i = 0; i < 4; i++) {
        displacements[i] -= base_address;
    }

    MPI_Type_create_struct(4, block_lengths, displacements, types, &MPI_Edge);
    MPI_Type_commit(&MPI_Edge);
}

static void partition_graph(graph_t* G) {
    vector<int>      rowsCount(size_), rowsDisp(size_);
    vector<int>      edgesCount(size_), edgesDisp(size_);
    vector<edge_id_t> globalRows;
    vector<vertex_id_t> globalEndV;
    vector<weight_t>    globalWeights;
    vector<edge_id_t>   globalEdgeIds;

    if (rank_ == 0) {
        globalRows.assign(G->rowsIndices, G->rowsIndices + TotVertices + 1);
        globalEndV.assign(G->endV,       G->endV + G->rowsIndices[TotVertices]);
        globalWeights.assign(G->weights, G->weights  + G->rowsIndices[TotVertices]);
        globalEdgeIds.resize(G->rowsIndices[TotVertices]);
        for (edge_id_t j = 0; j < globalEdgeIds.size(); j++)
            globalEdgeIds[j] = j;

        for (int p = 0; p < size_; ++p) {
            uint32_t start = (TotVertices * p) / size_;
            uint32_t end   = (TotVertices * (p + 1)) / size_;
            rowsCount[p]  = static_cast<int>(end - start + 1);
            rowsDisp[p]   = static_cast<int>(start);
            edgesCount[p] = static_cast<int>(globalRows[end] - globalRows[start]);
            edgesDisp[p]  = static_cast<int>(globalRows[start]);
        }
    }

    int localRowsCount = 0, localEdgesCount = 0;
    MPI_Scatter(rowsCount.data(),  1, MPI_INT,
                &localRowsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(edgesCount.data(), 1, MPI_INT,
                &localEdgesCount,1, MPI_INT, 0, MPI_COMM_WORLD);
    local_n = localRowsCount - 1;

    edge_id_t*   localRows     = (edge_id_t*)malloc(localRowsCount * sizeof(edge_id_t));
    vertex_id_t* localEndV     = (vertex_id_t*)malloc(localEdgesCount * sizeof(vertex_id_t));
    weight_t*    localWeights  = (weight_t*)malloc(localEdgesCount * sizeof(weight_t));
    edge_id_t*   localEdgeIds  = (edge_id_t*)malloc(localEdgesCount * sizeof(edge_id_t));

    MPI_Scatterv(
        rank_ == 0 ? G->rowsIndices : nullptr,
        rowsCount.data(), rowsDisp.data(), MPI_UINT64_T,
        localRows,         localRowsCount,    MPI_UINT64_T,
        0, MPI_COMM_WORLD
    );
    MPI_Scatterv(
        rank_ == 0 ? G->endV : nullptr,
        edgesCount.data(), edgesDisp.data(), MPI_UINT32_T,
        localEndV,          localEdgesCount,   MPI_UINT32_T,
        0, MPI_COMM_WORLD
    );
    MPI_Scatterv(
        rank_ == 0 ? G->weights : nullptr,
        edgesCount.data(), edgesDisp.data(), MPI_DOUBLE,
        localWeights,       localEdgesCount,   MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    MPI_Scatterv(
        rank_ == 0 ? globalEdgeIds.data() : nullptr,
        edgesCount.data(), edgesDisp.data(), MPI_UINT64_T,
        localEdgeIds,       localEdgesCount,   MPI_UINT64_T,
        0, MPI_COMM_WORLD
    );

    edge_id_t base = localRows[0];
    for (int i = 0; i < localRowsCount; ++i)
        localRows[i] -= base;

    uint32_t start_global = (TotVertices * rank_) / size_;
    local2global = (vertex_id_t*)malloc(local_n * sizeof(vertex_id_t));
    for (int i = 0; i < local_n; ++i)
        local2global[i] = start_global + i;

    if (rank_ == 0) {
        free(G->rowsIndices);
        free(G->endV);
        free(G->weights);
    }
    G->rowsIndices   = localRows;
    G->endV          = localEndV;
    G->weights       = localWeights;
    localEdgeIdArr   = localEdgeIds;
}

static int find_comp(vector<int> &parent, int i) {
    return parent[i] == i ? i : (parent[i] = find_comp(parent, parent[i]));
}

static void union_comp(vector<int> &parent, int x, int y) {
    int rx = find_comp(parent, x);
    int ry = find_comp(parent, y);
    if (rx != ry) parent[ry] = rx;
}

static vector<Edge> findLocalMinEdges(graph_t *G, vector<int> &parent, int &localEdgesCount) {
    vector<Edge> minEdgeForComp(TotVertices);
    for (uint32_t i = 0; i < TotVertices; ++i) {
        minEdgeForComp[i].weight  = numeric_limits<weight_t>::max();
        minEdgeForComp[i].startV  = (vertex_id_t)-1;
        minEdgeForComp[i].endV    = (vertex_id_t)-1;
        minEdgeForComp[i].edge_id = (edge_id_t)-1;
    }
    for (int u = 0; u < local_n; ++u) {
        vertex_id_t gu = local2global[u];
        int comp_u     = find_comp(parent, gu);
        for (edge_id_t idx = G->rowsIndices[u]; idx < G->rowsIndices[u + 1]; ++idx) {
            vertex_id_t v = G->endV[idx];
            weight_t    w = G->weights[idx];
            int comp_v   = find_comp(parent, v);
            if (comp_u != comp_v && w < minEdgeForComp[comp_u].weight) {
                minEdgeForComp[comp_u].weight  = w;
                minEdgeForComp[comp_u].startV  = gu;
                minEdgeForComp[comp_u].endV    = v;
                minEdgeForComp[comp_u].edge_id = localEdgeIdArr[idx];
            }
        }
    }
    vector<Edge> localEdges;
    localEdges.reserve(local_n);
    for (uint32_t c = 0; c < TotVertices; ++c) {
        if (minEdgeForComp[c].startV != (vertex_id_t)-1)
            localEdges.push_back(minEdgeForComp[c]);
    }
    localEdgesCount = (int)localEdges.size();
    return localEdges;
}

extern "C" void init_mst(graph_t *G) {
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    create_mpi_edge_type();

    if (rank_ == 0) {
        TotVertices = G->n;
    }
    MPI_Bcast(&TotVertices, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d sees TotVertices = %u\n", rank_, TotVertices);
    fflush(stdout);

    partition_graph(G);

    if (rank_ != 0) {
        freopen("/dev/null", "w", stdout);
        freopen("/dev/null", "w", stderr);
    }

    G->rank  = rank_;
    G->nproc = size_;
    lgsize = 0;
    while ((1 << lgsize) < size_) ++lgsize;
}


extern "C" void* MST(graph_t *G) {
    vector<int> parent(TotVertices);
    for (uint32_t i = 0; i < TotVertices; ++i)
        parent[i] = i;
    vector<Edge> mstEdges;
    mstEdges.reserve(TotVertices - 1);

    bool somethingMerged = true;
    while (somethingMerged) {
        int localEdgesCount = 0;
        vector<Edge> localEdges = findLocalMinEdges(G, parent, localEdgesCount);

        vector<int> recvCounts(size_);
        MPI_Gather(&localEdgesCount, 1, MPI_INT,
                   recvCounts.data(),    1, MPI_INT,
                   0, MPI_COMM_WORLD);

        vector<Edge> globalEdges;
        int totalCount = 0;
        if (rank_ == 0) {
            for (int c : recvCounts)
                totalCount += c;
            globalEdges.resize(totalCount);

            for (int i = 0; i < localEdgesCount; ++i)
                globalEdges[i] = localEdges[i];
            int offset = localEdgesCount;
            for (int p = 1; p < size_; ++p) {
                int cnt = recvCounts[p];
                if (cnt > 0) {
                    MPI_Recv(&globalEdges[offset], cnt, MPI_Edge,
                             p, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                offset += cnt;
            }
        } else {
            if (localEdgesCount > 0) {
                MPI_Send(localEdges.data(), localEdgesCount,
                         MPI_Edge, 0, 777, MPI_COMM_WORLD);
            }
        }

        bool merged = false;
        if (rank_ == 0) {
            vector<Edge> best(TotVertices);
            for (uint32_t c = 0; c < TotVertices; ++c) {
                best[c].weight  = numeric_limits<weight_t>::max();
                best[c].startV  = (vertex_id_t)-1;
                best[c].endV    = (vertex_id_t)-1;
                best[c].edge_id = (edge_id_t)-1;
            }

            for (const Edge &e : globalEdges) {
                int comp = find_comp(parent, e.startV);
                if (e.weight < best[comp].weight) {
                    best[comp] = e;
                }
            }

            for (uint32_t c = 0; c < TotVertices; ++c) {
                if (best[c].startV != (vertex_id_t)-1) {
                    int cu = find_comp(parent, best[c].startV);
                    int cv = find_comp(parent, best[c].endV);
                    if (cu != cv) {
                        union_comp(parent, cu, cv);
                        mstEdges.push_back(best[c]);
                        merged = true;
                    }
                }
            }
        }

        MPI_Bcast(&merged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Bcast(parent.data(), TotVertices, MPI_INT, 0, MPI_COMM_WORLD);

        somethingMerged = merged;
    }

    return new vector<Edge>(std::move(mstEdges));
}


extern "C" void convert_to_output(graph_t *G, void* result, forest_t *out) {
    if (G->rank != 0) {
        delete reinterpret_cast<vector<Edge>*>(result);
        return;
    }
    
    vector<Edge> &mst = *reinterpret_cast<vector<Edge>*>(result);
    out->numTrees    = 1;
    out->numEdges    = mst.size();
    out->edge_id     = (edge_id_t*)malloc(mst.size() * sizeof(edge_id_t));
    out->p_edge_list = (edge_id_t*)malloc(2 * sizeof(edge_id_t));
    out->p_edge_list[0] = 0;
    out->p_edge_list[1] = (edge_id_t)mst.size();
    for (size_t i = 0; i < mst.size(); ++i)
        out->edge_id[i] = mst[i].edge_id;
    delete reinterpret_cast<vector<Edge>*>(result);
}

extern "C" void finalize_mst(graph_t *G) {
    int flag = 0;
    MPI_Finalized(&flag);
    if (!flag) {
        MPI_Type_free(&MPI_Edge);
        MPI_Finalize();
    }
}
