/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright 2015 Red Hat, Inc.
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (scale, _("Scale"), 1.0)
    description(_("Scale, strength of effect"))
    value_range (0.0, 1.0)

property_boolean (srgb, _("sRGB"), TRUE)
    description (_("Use sRGB gamma instead of linear"))

#else

#define GEGL_OP_POINT_FILTER
#define GEGL_OP_C_SOURCE sepia.c

#include "gegl-op.h"

static void prepare (GeglOperation *operation)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);
  const Babl *format;

  if (o->srgb)
    format = babl_format ("R'G'B'A float");
  else
    format = babl_format ("RGBA float");

  gegl_operation_set_format (operation, "input", format);
  gegl_operation_set_format (operation, "output", format);
}

static gboolean
process (GeglOperation       *operation,
         void                *in_buf,
         void                *out_buf,
         glong                n_pixels,
         const GeglRectangle *roi,
         gint                 level)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);
  gfloat *in = in_buf;
  gfloat *out = out_buf;
  gfloat m[25] = { 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 1.0 };
  glong i;

  m[0] = 0.393 + 0.607 * (1.0 - o->scale);
  m[1] = 0.769 - 0.769 * (1.0 - o->scale);
  m[2] = 0.189 - 0.189 * (1.0 - o->scale);

  m[5] = 0.349 - 0.349 * (1.0 - o->scale);
  m[6] = 0.686 + 0.314 * (1.0 - o->scale);
  m[7] = 0.168 - 0.168 * (1.0 - o->scale);

  m[10] = 0.272 - 0.272 * (1.0 - o->scale);
  m[11] = 0.534 - 0.534 * (1.0 - o->scale);
  m[12] = 0.131 + 0.869 * (1.0 - o->scale);

  for (i = 0; i < n_pixels; i++)
    {
      out[0] = m[0] * in[0] + m[1] * in[1] + m[2] * in[2];
      out[1] = m[5] * in[0] + m[6] * in[1] + m[7] * in[2];
      out[2] = m[10] * in[0] + m[11] * in[1] + m[12] * in[2];
      out[3] = in[3];

      in += 4;
      out += 4;
    }

  return TRUE;
}

#include "opencl/sepia.cl.h"

static GeglClRunData *cl_data = NULL;

static gboolean
cl_process (GeglOperation       *op,
            cl_mem               in_tex,
            cl_mem               out_tex,
            const size_t         samples,
            const GeglRectangle *roi,
            gint                 level)
{
  if(!cl_data)
    {
      const char *kernel_name[] = {"gegl_sepia", NULL};
      cl_data = gegl_cl_compile_and_build(sepia_cl_source, kernel_name);
    }

  if(!cl_data)
    return TRUE;
  else
    {
      GeglProperties *o = GEGL_PROPERTIES (op);
      cl_int cl_err = 0;
      gfloat m[12] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      m[0] = 0.393 + 0.607 * (1.0 - o->scale);
      m[1] = 0.769 - 0.769 * (1.0 - o->scale);
      m[2] = 0.189 - 0.189 * (1.0 - o->scale);

      m[4] = 0.349 - 0.349 * (1.0 - o->scale);
      m[5] = 0.686 + 0.314 * (1.0 - o->scale);
      m[6] = 0.168 - 0.168 * (1.0 - o->scale);

      m[8] = 0.272 - 0.272 * (1.0 - o->scale);
      m[9] = 0.534 - 0.534 * (1.0 - o->scale);
      m[10] = 0.131 + 0.869 * (1.0 - o->scale);

      cl_mem coefs = gegl_clCreateBuffer(gegl_cl_get_context(),
                                         CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                         12 * sizeof(cl_float), m, &cl_err);
      CL_CHECK;
      const size_t global_ws[2] = {samples, 1};

      cl_err = gegl_cl_set_kernel_args(cl_data->kernel[0],
                                       sizeof(cl_mem), (void *)&in_tex,
                                       sizeof(cl_mem), (void *)&out_tex,
                                       sizeof(cl_mem), (void *)&coefs,
                                       NULL);

      cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue(),
                                           cl_data->kernel[0], 2, NULL,
                                           global_ws, NULL, 0, NULL, NULL);
      CL_CHECK;

      cl_err = gegl_clFinish(gegl_cl_get_command_queue());
      CL_CHECK;

      return FALSE;

    error:
      return TRUE;
    }
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass *operation_class = GEGL_OPERATION_CLASS (klass);
  GeglOperationPointFilterClass *point_filter_class =
    GEGL_OPERATION_POINT_FILTER_CLASS (klass);

  operation_class->prepare = prepare;
  point_filter_class->process = process;

  point_filter_class->cl_process = cl_process;
  operation_class->opencl_support = TRUE;

  gegl_operation_class_set_keys (operation_class,
                                 "name"       , "gegl:sepia",
                                 "title",       _("Sepia"),
                                 "categories" , "color",
                                 "description", _("Converts the input image to sepia"),
                                 NULL);
}

#endif
