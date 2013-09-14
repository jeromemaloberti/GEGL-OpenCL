static const char* box_min_cl_source =
"__kernel void kernel_min_hor (__global const float4     *in,                  \n"
"                              __global       float4     *aux,                 \n"
"                              int width, int radius)                          \n"
"{                                                                             \n"
"  const int in_index = get_global_id(0) * (width + 2 * radius)                \n"
"                       + (radius + get_global_id (1));                        \n"
"                                                                              \n"
"  const int aux_index = get_global_id(0) * width + get_global_id (1);         \n"
"  int i;                                                                      \n"
"  float4 min;                                                                 \n"
"  float4 in_v;                                                                \n"
"                                                                              \n"
"  min = (float4)(FLT_MAX);                                                    \n"
"                                                                              \n"
"  if (get_global_id(1) < width)                                               \n"
"    {                                                                         \n"
"      for (i=-radius; i <= radius; i++)                                       \n"
"        {                                                                     \n"
"          in_v = in[in_index + i];                                            \n"
"          min = min > in_v ? in_v : min;                                      \n"
"        }                                                                     \n"
"        aux[aux_index] = min;                                                 \n"
"    }                                                                         \n"
"}                                                                             \n"
"                                                                              \n"
"__kernel void kernel_min_ver (__global const float4     *aux,                 \n"
"                              __global       float4     *out,                 \n"
"                              int width, int radius)                          \n"
"{                                                                             \n"
"                                                                              \n"
"  const int out_index = get_global_id(0) * width + get_global_id (1);         \n"
"  int aux_index = out_index;                                                  \n"
"  int i;                                                                      \n"
"  float4 min;                                                                 \n"
"  float4 aux_v;                                                               \n"
"                                                                              \n"
"  min = (float4)(FLT_MAX);                                                    \n"
"                                                                              \n"
"  if(get_global_id(1) < width)                                                \n"
"    {                                                                         \n"
"      for (i=-radius; i <= radius; i++)                                       \n"
"        {                                                                     \n"
"          aux_v = aux[aux_index];                                             \n"
"          min = min > aux_v ? aux_v : min;                                    \n"
"          aux_index += width;                                                 \n"
"        }                                                                     \n"
"        out[out_index] = min;                                                 \n"
"    }                                                                         \n"
"}                                                                             \n"
;