__kernel void gegl_sepia	(__global const float4     *in,
                           __global       float4     *out,
													 __constant     float4     *coef)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v;

/*
	float4 tmp = coef[0] * in_v;
	out_v.x = tmp.x + tmp.y + tmp.z;
	
	tmp = coef[1] * in_v;
	out_v.y = tmp.x + tmp.y + tmp.z;
	
	tmp = coef[2] * in_v;
	out_v.z = tmp.x + tmp.y + tmp.z;
	*/
	out_v.x = dot(in_v, coef[0]);
	out_v.y = dot(in_v, coef[1]);
	out_v.z = dot(in_v, coef[2]);
  out_v.w   =  in_v.w;

	out[gid]  =  out_v;
}
