#ifndef FLOCKING_ENV_H
#define FLOCKING_ENV_H

#include <vector>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

using Matrix = std::vector<std::vector<double>>;

struct Env {
    // Define env properties here
    // Example: position, velocity, heading, etc.
    Matrix p; // Position
    Matrix dp; // Velocity
    Matrix obs;
    std::vector<double> leader_state;
    double d_sen;
    int topo_nei_max;
    int n_a; // Number of agents
    int obs_dim_agent; // Observation dimension for each env
    int dim; // Dimensionality
    // Add other required properties here
};

void _get_observation(double *p_input, 
                      double *dp_input, 
                      double *heading_input,
                      double *obs_input, 
                      double *leader_state_input, 
                      int *neighbor_index_input, 
                      int *leader_index_input,
                      int *random_permutation,
                      double *boundary_pos_input, 
                      double d_sen, 
                      int topo_nei_max, 
                      int n_a, 
                      int n_l,
                      int obs_dim_agent, 
                      int dim, 
                      bool *condition);
void _get_reward(double *p_input, 
                 double *dp_input,
                 double *heading_input, 
                 double *act_input, 
                 double *reward_input, 
                 double *leader_state_input, 
                 int *neighbor_index_input,
                 int *leader_index_input,
                 double *boundary_pos_input, 
                 double d_sen, 
                 double d_ref, 
                 int topo_nei_max, 
                 int n_a, 
                 int n_l,
                 int dim, 
                 bool *condition, 
                 bool *is_collide_b2b_input, 
                 bool *is_collide_b2w_input, 
                 double *coefficients);
std::tuple<Matrix, Matrix, std::vector<int>> _get_focused(Matrix Pos, 
                                                          Matrix Vel, 
                                                          double norm_threshold, 
                                                          int width, 
                                                          bool remove_self);
void _sf_b2b_all(double *p_input,
                 double *sf_b2b_input, 
                 double *d_b2b_edge_input,
                 bool *is_collide_b2b_input,
                 double *boundary_pos_input,
                 double *d_b2b_center_input,
                 int n_a,
                 int dim,
                 double k_ball,
                 bool is_periodic);
void _make_periodic(Matrix& x, 
                    double L, 
                    std::vector<double> bound_pos, 
                    bool is_rel);
void _get_dist_b2w(double *p_input, 
                   double *r_input, 
                   double *d_b2w_input, 
                   bool *isCollision_input, 
                   int dim, 
                   int n_a, 
                   double *boundary_L_half);
Matrix _concatenate(const Matrix& arr1, const Matrix& arr2, int axis);
Matrix _extract_column(const Matrix& arr, size_t col_index);
std::vector<double> _extract_column_one(const Matrix& arr, size_t col_index);
double _norm(std::vector<double>& v);
bool _all_elements_greater_than_(std::vector<int>& arr, int n_l);
double _rho_cos_dec(double z, double delta, double r);
Matrix _vector_to_matrix(const std::vector<double>& vec);
std::vector<double> _matrix_to_vector(const Matrix& matrix);

#ifdef __cplusplus
}
#endif

#endif
