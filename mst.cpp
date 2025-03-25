#include "defs.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <mpi.h>
#include <cassert>

using namespace std;

int rank_, size_, lgsize;
uint32_t TotVertices;

struct Edge {
    vertex_id_t startV;
    vertex_id_t endV;
    weight_t weight;
    edge_id_t edge_id;
};

MPI_Datatype MPI_Edge;

void create_mpi_edge_type() {
    Edge dummy = {};
    int block_lengths[4] = {1, 1, 1, 1};
    MPI_Aint displacements[4];
    MPI_Datatype types[4] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_FLOAT, MPI_UNSIGNED};

    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.startV, &displacements[0]);
    MPI_Get_address(&dummy.endV, &displacements[1]);
    MPI_Get_address(&dummy.weight, &displacements[2]);
    MPI_Get_address(&dummy.edge_id, &displacements[3]);

    for (int i = 0; i < 4; i++) {
        displacements[i] -= base_address;
    }

    MPI_Type_create_struct(4, block_lengths, displacements, types, &MPI_Edge);
    MPI_Type_commit(&MPI_Edge);
}


extern "C" void init_mst(graph_t *G)
{
    int flag = 0;
    MPI_Initialized(&flag);
    if(!flag) {
        MPI_Init(NULL, NULL);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    create_mpi_edge_type();

    if (rank_ != 0) {
        FILE* null_out = freopen("/dev/null", "w", stdout);
        FILE* null_err = freopen("/dev/null", "w", stderr);
        (void)null_out;
        (void)null_err;
    }


    TotVertices = G->n;
    G->rank = rank_;
    G->nproc = size_;

    lgsize = 0;
    while ((1 << lgsize) < size_) {
        lgsize++;
    }
}

static int find_comp(vector<int> &parent, int i)
{
    if (parent[i] == i) return i;
    parent[i] = find_comp(parent, parent[i]);
    return parent[i];
}

static void union_comp(vector<int> &parent, int x, int y)
{
    int rx = find_comp(parent, x);
    int ry = find_comp(parent, y);
    if (rx != ry) {
        parent[ry] = rx;
    }
}

static vector<Edge> findLocalMinEdges(graph_t *G,
                                      vector<int> &component,
                                      int &localEdgesCount)
{
  
    vector<Edge> minEdgeForComp(G->n);

    for (int i = 0; i < (int)G->n; i++) {
        minEdgeForComp[i].weight  = numeric_limits<weight_t>::max();
        minEdgeForComp[i].startV  = (vertex_id_t)-1;
        minEdgeForComp[i].endV    = (vertex_id_t)-1;
        minEdgeForComp[i].edge_id = (edge_id_t)-1;
    }

   
    for (vertex_id_t u = 0; u < G->n; ++u) {
        int comp_u = find_comp(component, u);
        for (edge_id_t j = G->rowsIndices[u]; j < G->rowsIndices[u + 1]; ++j) {
            vertex_id_t v = G->endV[j];
            weight_t w = G->weights[j];
            int comp_v = find_comp(component, v);

            if (comp_u != comp_v) {
    
                if (w < minEdgeForComp[comp_u].weight) {
                    minEdgeForComp[comp_u].weight  = w;
                    minEdgeForComp[comp_u].startV  = u;
                    minEdgeForComp[comp_u].endV    = v;
                    minEdgeForComp[comp_u].edge_id = j;
                }
            }
        }
    }

    vector<Edge> localEdges;
    localEdges.reserve(G->n);
    for (int i = 0; i < (int)G->n; i++) {
        if (minEdgeForComp[i].startV != (vertex_id_t)-1) {
            localEdges.push_back(minEdgeForComp[i]);
        }
    }
    localEdgesCount = (int)localEdges.size();
    return localEdges;
}

extern "C" void* MST(graph_t *G)
{

    vector<int> parent(G->n);
    for (vertex_id_t i = 0; i < G->n; i++) {
        parent[i] = i;
    }

    vector<Edge> mstEdges;
    mstEdges.reserve(G->n - 1); 

    bool somethingMerged = true;

    while (somethingMerged)
    {
        int localEdgesCount = 0;
        vector<Edge> localEdges = findLocalMinEdges(G, parent, localEdgesCount);

        vector<int> recvCounts(size_);

        MPI_Gather(&localEdgesCount, 1, MPI_INT,
                   &recvCounts[0], 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        vector<Edge> globalEdges;
        int totalCount = 0;

        if (rank_ == 0) {
            for (int i = 0; i < size_; i++) {
                totalCount += recvCounts[i];
            }
            globalEdges.resize(totalCount);

            for (int i = 0; i < localEdgesCount; i++) {
                globalEdges[i] = localEdges[i];
            }
            int offset = localEdgesCount;

            for (int proc = 1; proc < size_; proc++) {
                int count = recvCounts[proc];
                if (count > 0) {
             
                    MPI_Recv(&globalEdges[offset], count, MPI_Edge, proc, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                }
                offset += count;
            }
        }
        else {
            if (localEdgesCount > 0) {
             
                MPI_Send(localEdges.data(), localEdgesCount, MPI_Edge, 0, 777, MPI_COMM_WORLD);

            }
        }

        bool localSomethingMerged = false;
        if (rank_ == 0) {
    
            for (int i = 0; i < totalCount; i++) {
                Edge &e = globalEdges[i];
                int compU = find_comp(parent, e.startV);
                int compV = find_comp(parent, e.endV);
                if (compU != compV) {
                 
                    mstEdges.push_back(e);
                    union_comp(parent, compU, compV);
                    localSomethingMerged = true;
                }
            }
        }

        MPI_Bcast(&localSomethingMerged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        MPI_Bcast(parent.data(), parent.size(), MPI_INT, 0, MPI_COMM_WORLD);

        somethingMerged = localSomethingMerged;
    }

    auto *mstResult = new vector<Edge>(std::move(mstEdges));
    return mstResult;
}

extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output)
{
    vector<Edge> &mstEdges = *reinterpret_cast<vector<Edge>*>(result);

    trees_output->numTrees = 1;
    trees_output->numEdges = mstEdges.size();

    trees_output->edge_id = (edge_id_t*)malloc(mstEdges.size() * sizeof(edge_id_t));
    trees_output->p_edge_list = (edge_id_t *)malloc(2 * sizeof(edge_id_t));

    trees_output->p_edge_list[0] = 0;                
    trees_output->p_edge_list[1] = mstEdges.size();   

    for (size_t i = 0; i < mstEdges.size(); i++) {
        trees_output->edge_id[i] = mstEdges[i].edge_id;
    }

    delete reinterpret_cast<vector<Edge>*>(result);
}

extern "C" void finalize_mst(graph_t *G)
{
    int flag = 0;
    MPI_Finalized(&flag);
    if(!flag) {
        MPI_Type_free(&MPI_Edge);
        MPI_Finalize();
    }
}
