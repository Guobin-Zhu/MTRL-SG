#ifndef ADVERSARIAL_ENV_ENV_H
#define ADVERSARIAL_ENV_ENV_H

#include <vector>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

using Matrix = std::vector<std::vector<double>>;

void _get_observation(double *p_input, 
                      double *dp_input, 
                      double *heading_input,
                      double *obs_input,
                      double *boundary_pos_input, 
                      double *hp_input,
                      int *index_l_input,
                      int *index_r_input,
                      double d_sen_l, 
                      double d_sen_r,
                      double hp_l_max,
                      double hp_r_max,
                      int topo_nei_l2l, 
                      int topo_nei_l2r,
                      int topo_nei_r2l, 
                      int topo_nei_r2r,
                      int n_l, 
                      int n_r,
                      int n_lr_init,
                      int obs_dim_agent, 
                      int dim, 
                      bool *condition);
void _get_reward(double *act_input, 
                 double *boundary_pos_input, 
                 double *hp_input,
                 double *reward_l_input,
                 double *reward_r_input, 
                 double *coefficients,
                 bool *condition, 
                 bool *is_collide_b2b_input, 
                 bool *is_collide_b2w_input,
                 int *index_l_input,
                 int *index_r_input,
                 int *attack_nei_index_input,
                 int *safe_nei_index_input,
                 int attack_max, 
                 int safe_max,
                 int n_l, 
                 int n_r,
                 int n_lr_init,
                 int dim);
void _process_act(int *index_l_last_input,
                  int *index_r_last_input,
                  int *index_l_input,
                  int *index_r_input,
                  double *a_com_input,
                  double *a_true_input,
                  double *a_input,
                  int dim,
                  int n_lr,
                  int n_lr_init,
                  int n_l_last,
                  int n_r_last,
                  int n_l,
                  int n_r);
void _process_attack(int *index_l_input,
                     int *index_r_input,
                     int *dead_index_input,
                     int *attack_neigh_input,
                     int *safe_neigh_input,
                     double *p_input,
                     double *dp_input,
                     double *hp_input,
                     double *boundary_pos_input,
                     bool *is_training,
                     bool is_periodic,
                     double attack_radius,
                     double attack_angle,
                     double attack_hp,
                     double recover_hp,
                     int attack_max,
                     int safe_max,
                     int n_lr_init,
                     int n_l,
                     int n_r,
                     int dim);
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
                 int *dead_index_input,
                 int n_lr_init,
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
                   int n_pe, 
                   double *boundary_L_half);
Matrix _concatenate(const Matrix& arr1, const Matrix& arr2, int axis);
Matrix _extract_column(const Matrix& arr, size_t col_index);
std::vector<double> _extract_column_one(const Matrix& arr, size_t col_index);
double _dot_product(const std::vector<double>& a, const std::vector<double>& b);
double _norm(std::vector<double>& v);
double _rho_cos_dec(double z, double delta, double r);
Matrix _vector_to_matrix(const std::vector<double>& vec);
std::vector<double> _matrix_to_vector(const Matrix& matrix);
std::vector<size_t> _intersect_and_sort(const std::vector<size_t>& v1, const std::vector<size_t>& v2);
std::vector<size_t> _intersect(const std::vector<size_t>& v1, const std::vector<size_t>& v2);

#ifdef __cplusplus
}
#endif

#endif
