#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
#include "runtime.Kokkos/Extensions.hpp"

#include <MathExtensions.hpp>

#include <KokkosBlas.hpp>

#include <StdExtensions.hpp>

#include <Print.hpp>

//#include <Algebra/Eigenvalue.hpp>

// KOKKOS_NET_API_EXTERNC void* Shepard2dSingle(void* xd_rcp_view_ptr, void* zd_rcp_view_ptr, const float& p, void* xi_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept;
//
// KOKKOS_NET_API_EXTERNC void* Shepard2dDouble(void* xd_rcp_view_ptr, void* zd_rcp_view_ptr, const double& p, void* xi_rcp_view_ptr, const ExecutionSpaceKind& execution_space)
// noexcept;

// struct coordinate
//{
//    coord_type type;
//
//    double x;
//
//    double xlow;
//    double xhigh;
//
//    double y;
//
//    double ylow;
//    double yhigh;
//
//    double z;
//};

namespace Cartography
{
    enum coord_type
    {
        INRANGE       = 0,
        OUTRANGE      = 1,
        UNDEFINED     = 2,
        EXCLUDEDRANGE = 3
    };

    enum en_contour_kind
    {
        /* Method of drawing the contour lines found */
        CONTOUR_KIND_LINEAR,
        CONTOUR_KIND_CUBIC_SPL,
        CONTOUR_KIND_BSPLINE
    };

    enum en_contour_levels_kind
    {
        /* How contour levels are set */
        LEVELS_AUTO, /* automatically selected */
        LEVELS_INCREMENTAL, /* user specified start & increment */
        LEVELS_DISCRETE /* user specified discrete levels */
    };

    enum DimensionIndex : uint8
    {
        X                   = 0,
        Y                   = 1,
        Z                   = 2,
        DimensionIndexCount = 3
    };

    enum CoordinateIndex : uint8
    {
        Value                = 0,
        Low                  = 1,
        Type                 = 1,
        High                 = 2,
        CoordinateIndexCount = 3
    };

    template<typename DataType, class ExecutionSpace>
    using CoordinateVector = Kokkos::View<DataType* [DimensionIndexCount][CoordinateIndexCount], typename ExecutionSpace::array_layout, ExecutionSpace>;

    template<typename DataType, class ExecutionSpace, size_type M>
    using PointVector = Kokkos::View<DataType* [M], typename ExecutionSpace::array_layout, ExecutionSpace>;

    // tri-diag matrix
    typedef double tri_diag[3];

    template<typename DataType, class ExecutionSpace>
    struct Contours
    {
        Contours<DataType, ExecutionSpace>* next;

        CoordinateVector<DataType, ExecutionSpace> coords;

        char isNewLevel;

        double z;
    };

    template<typename DataType, class ExecutionSpace>
    struct IsoCurve
    {
        IsoCurve<DataType, ExecutionSpace>* next;
        /* how many points are allocated */
        int p_max;
        /* count of points in points */
        int                                        p_count;
        CoordinateVector<DataType, ExecutionSpace> points;
    };

    enum position_type
    {
        first_axes,
        second_axes,
        graph,
        screen,
        character,
        polar_axes
    };

    struct position
    {
        position_type scalex;
        position_type scaley;
        position_type scalez;
        double        x;
        double        y;
        double        z;
    };

    struct surface_points
    {
        surface_points*  next_sp; /* pointer to next plot in linked list */
        int              token; /* last token used, for second parsing pass */
        PLOT_TYPE        plot_type; /* DATA2D? DATA3D? FUNC2D FUNC3D? NODATA? */
        PLOT_STYLE       plot_style; /* style set by "with" or by default */
        char*            title; /* plot title, a.k.a. key entry */
        position*        title_position; /* title at {beginning|end|<xpos>,<ypos>} */
        bool             title_no_enhanced; /* don't typeset title in enhanced mode */
        bool             title_is_automated; /* TRUE if title was auto-generated */
        bool             title_is_suppressed; /* TRUE if 'notitle' was specified */
        bool             noautoscale; /* ignore data from this plot during autoscaling */
        lp_style_type    lp_properties;
        arrow_style_type arrow_properties;
        fill_style_type  fill_properties;
        text_label*      labels; /* Only used if plot_style == LABELPOINTS */
        t_image          image_properties; /* only used if plot_style is IMAGE, RGBIMAGE or RGBA_IMAGE */
        udvt_entry*      sample_var; /* used by '+' if plot has private sampling range */
        udvt_entry*      sample_var2; /* used by '++' if plot has private sampling range */

        /* 2D and 3D plot structure fields overlay only to this point */

        PLOT_SMOOTH plot_smooth; /* EXPERIMENTAL: smooth lines in 3D */
        bool        opt_out_of_hidden3d; /* set by "nohidden" option to splot command */
        bool        opt_out_of_contours; /* set by "nocontours" option to splot command */
        bool        opt_out_of_surface; /* set by "nosurface" option to splot command */
        bool        pm3d_color_from_column;
        bool        has_grid_topology;
        int         hidden3d_top_linetype; /* before any calls to load_linetype() */
        int         iteration; /* needed for tracking iteration */

        vgrid* vgrid; /* used only for voxel plots */
        double iso_level; /* used only for voxel plots */

        /* Data files only - num of isolines read from file. For functions,  */
        /* num_iso_read is the number of 'primary' isolines (in x direction) */
        int               num_iso_read;
        gnuplot_contours* contours; /* NULL if not doing contours. */
        iso_curve*        iso_crvs; /* the actual data */
        char              pm3d_where[7]; /* explicitly given base, top, surface */
    };

    enum edge_position
    {
        INNER_MESH = 1,
        BOUNDARY,
        DIAGONAL
    };

    struct poly_struct;
    struct Coordinate;

    struct edge_struct
    {
        poly_struct*  poly[2]; /* Each edge belongs to up to 2 polygons */
        Coordinate*   vertex[2]; /* The two extreme points of this edge. */
        edge_struct*  next; /* To chain lists */
        bool          is_active; /* is edge is 'active' at certain Z level? */
        edge_position position; /* position of edge in mesh */
    };

    struct poly_struct
    {
        edge_struct* edge[3]; /* As we do triangulation here... */
        poly_struct* next; /* To chain lists. */
    };

    /* Contours are saved using this struct list. */
    struct cntr_struct
    {
        double       X;
        double       Y;
        cntr_struct* next; /* To chain lists. */
    };

    namespace Internal
    {
    }

    template<typename DataType, class ExecutionSpace>
    static Contours<DataType, ExecutionSpace>* Contour(int num_isolines, IsoCurve<DataType, ExecutionSpace> iso_lines[])
    {
        int                      i;
        int                      num_of_z_levels; /* # Z contour levels. */
        double*                  zlist;
        poly_struct*             p_polys;
        poly_struct*             p_poly;
        edge_struct*             p_edges;
        edge_struct*             p_edge;
        double                   z  = 0;
        double                   z0 = 0;
        double                   dz = 0;
        struct gnuplot_contours* save_contour_list;

        /* HBB FIXME 20050804: The number of contour_levels as set by 'set
         * cnrparam lev inc a,b,c' is almost certainly wrong if z axis is
         * logarithmic */
        num_of_z_levels = contour_levels;
        interp_kind     = contour_kind;

        contour_list = NULL;

        /*
         * Calculate min/max values :
         */
        calc_min_max(num_isolines, iso_lines, &x_min, &y_min, &z_min, &x_max, &y_max, &z_max);

        /*
         * Generate list of edges (p_edges) and list of triangles (p_polys):
         */
        gen_triangle(num_isolines, iso_lines, &p_polys, &p_edges);

        crnt_cntr_pt_index = 0;

        if(contour_levels_kind == LEVELS_AUTO)
        {
            if(nonlinear(&Z_AXIS))
            {
                z_max = eval_link_function(Z_AXIS.linked_to_primary, z_max);
                z_min = eval_link_function(Z_AXIS.linked_to_primary, z_min);
            }
            dz = fabs(z_max - z_min);
            if(dz == 0)
                return NULL; /* empty z range ? */
            /* Find a tic step that will generate approximately the
             * desired number of contour levels. The "* 2" is historical.
             * */
            dz              = quantize_normal_tics(dz, ((int)contour_levels + 1) * 2);
            z0              = floor(z_min / dz) * dz;
            num_of_z_levels = (int)floor((z_max - z0) / dz);
            if(num_of_z_levels <= 0)
                return NULL;
        }

        /* Build a list of contour levels */
        zlist = gp_alloc(num_of_z_levels * sizeof(double), NULL);
        for(i = 0; i < num_of_z_levels; i++)
        {
            switch(contour_levels_kind)
            {
                case LEVELS_AUTO:
                    z = z0 + (i + 1) * dz;
                    z = CheckZero(z, dz);
                    if(nonlinear(&Z_AXIS))
                        z = eval_link_function((&Z_AXIS), z);
                    break;
                case LEVELS_INCREMENTAL:
                    if(Z_AXIS.log)
                        z = contour_levels_list[0] * pow(contour_levels_list[1], (double)i);
                    else
                        z = contour_levels_list[0] + i * contour_levels_list[1];
                    break;
                case LEVELS_DISCRETE: z = contour_levels_list[i]; break;
            }
            zlist[i] = z;
        }
        /* Sort the list high-to-low if requested */
        if(contour_sortlevels)
            qsort(zlist, num_of_z_levels, sizeof(double), reverse_sort);

        /* Create contour line for each z value in the list */
        for(i = 0; i < num_of_z_levels; i++)
        {
            z                 = zlist[i];
            contour_level     = z;
            save_contour_list = contour_list;
            gen_contours(p_edges, z, x_min, x_max, y_min, y_max);
            if(contour_list != save_contour_list)
            {
                contour_list->isNewLevel = 1;
                /* Nov-2011 Use gprintf rather than sprintf so that LC_NUMERIC is used */
                gprintf(contour_list->label, sizeof(contour_list->label), contour_format, 1.0, z);
                contour_list->z = z;
            }
        }

        /* Free all contouring related temporary data. */
        free(zlist);
        while(p_polys)
        {
            p_poly = p_polys->next;
            free(p_polys);
            p_polys = p_poly;
        }
        while(p_edges)
        {
            p_edge = p_edges->next;
            free(p_edges);
            p_edges = p_edge;
        }

        return contour_list;
    }

}
