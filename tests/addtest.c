#include <glib-object.h>
#include "gegl.h"
#include "ctest.h"
#include "csuite.h"
#include "testutils.h"

#define SAMPLED_IMAGE_WIDTH 1 
#define SAMPLED_IMAGE_HEIGHT 1 

static GeglOp *input0, *input1;
static GeglOp *dest;


static void
test_add_g_object_new(Test *test)
{
  {
    GeglOp * add = g_object_new (GEGL_TYPE_ADD, NULL);  

    ct_test(test, GEGL_IS_ADD(add));
    ct_test(test, g_type_parent(GEGL_TYPE_ADD) == GEGL_TYPE_POINT_OP);
    ct_test(test, !strcmp("GeglAdd", g_type_name(GEGL_TYPE_ADD)));

    g_object_unref(add);
  }

  {
    GeglOp * add =  g_object_new (GEGL_TYPE_ADD, NULL);  

    ct_test(test, NULL == gegl_node_get_nth_input(GEGL_NODE(add), 0));
    ct_test(test, NULL == gegl_node_get_nth_input(GEGL_NODE(add), 1));
    ct_test(test, 2 == gegl_node_get_num_inputs(GEGL_NODE(add)));
    ct_test(test, 1 == gegl_node_get_num_outputs(GEGL_NODE(add)));

    g_object_unref(add);
  }

  {
    GeglOp * add = g_object_new (GEGL_TYPE_ADD, 
                                 "input0", input0,
                                 "input1", input1,
                                 NULL);  

    ct_test(test, input0 == (GeglOp*)gegl_node_get_nth_input(GEGL_NODE(add), 0));
    ct_test(test, input1 == (GeglOp*)gegl_node_get_nth_input(GEGL_NODE(add), 1));

    g_object_unref(add);
  }
}

static void
test_add_apply(Test *test)
{

  /* 
     add = input0 + input1 
     (.5,.7,.9) = (.1,.2,.3) + (.4,.5,.6)
  */

  GeglOp * add = g_object_new (GEGL_TYPE_ADD, 
                              "input0", input0,
                              "input1", input1,
                              NULL);  

  gegl_op_apply_image(add, dest, NULL); 

  ct_test(test, testutils_check_rgb_float_pixel(GEGL_IMAGE(dest), .1 + .4, .2 + .5, .3 + .6));  

  g_object_unref(add);
}

static void
add_test_setup(Test *test)
{
  GeglColorModel *rgb_float = gegl_color_model_instance("RgbFloat");


  input0 = testutils_rgb_float_sampled_image(SAMPLED_IMAGE_WIDTH, 
                                        SAMPLED_IMAGE_HEIGHT, 
                                        .1, .2, .3);

  input1 = testutils_rgb_float_sampled_image(SAMPLED_IMAGE_WIDTH, 
                                        SAMPLED_IMAGE_HEIGHT, 
                                        .4, .5, .6);

  dest = g_object_new (GEGL_TYPE_SAMPLED_IMAGE,
                       "colormodel", rgb_float,
                       "width", SAMPLED_IMAGE_WIDTH, 
                       "height", SAMPLED_IMAGE_HEIGHT,
                       NULL);  


  g_object_unref(rgb_float);
}

static void
add_test_teardown(Test *test)
{
  g_object_unref(input0);
  g_object_unref(input1);
  g_object_unref(dest);
}

Test *
create_add_test()
{
  Test* t = ct_create("GeglAddTest");

  g_assert(ct_addSetUp(t, add_test_setup));
  g_assert(ct_addTearDown(t, add_test_teardown));
  g_assert(ct_addTestFun(t, test_add_g_object_new));
  g_assert(ct_addTestFun(t, test_add_apply));

  return t; 
}
