/* This file is part of GEGL editor -- a gtk frontend for GEGL
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 * Copyright (C) 2003, 2004, 2006 Øyvind Kolås
 */

#ifndef EDITOR_H
#define EDITOR_H

#include "gegl-view.h"

typedef struct Editor 
{
  GtkWidget *window;
  GtkWidget *property_editor;
  GtkWidget *tree_editor;

  GtkWidget *structure;
  GtkWidget *property_pane;
  
  GtkWidget *graph_editor;
  GtkWidget *drawing_area;
  gchar     *composition_path;
  GeglNode  *gegl;
  gint       x;
  gint       y;
} Editor;

extern Editor editor;
GtkWidget * StockIcon (const gchar *id, GtkIconSize size, GtkWidget *widget);

#endif
