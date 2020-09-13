
#include "Analyzes/SpatialMethods.hpp"

#include <math.h>

void* NearestNeighborSingle(void* latlongdegrees_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            using ExecutionSpace = Kokkos::Serial;
            using LineVector     = Kokkos::View<float***, ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<float*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<LineVector>* latlongdegrees_rcp_view = reinterpret_cast<Teuchos::RCP<LineVector>*>(latlongdegrees_rcp_view_ptr);

            LineVector latlongdegrees = *(*latlongdegrees_rcp_view);

            Vector neighbors = Spatial::NearestNeighbor<float, ExecutionSpace>(latlongdegrees);

            typedef ViewBuilder<DataTypeKind::Single, 1, ExecutionSpaceKind::Serial>::ViewType view_type;
            view_type*                                                                         neighbors_view_ptr = new view_type("neighbors", neighbors.extent(0));

            for(size_type i0 = 0; i0 < neighbors.extent(0); ++i0)
            {
                (*neighbors_view_ptr)(i0) = neighbors(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(neighbors_view_ptr);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            using ExecutionSpace = Kokkos::OpenMP;
            using LineVector     = Kokkos::View<float***, ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<float*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<LineVector>* latlongdegrees_rcp_view = reinterpret_cast<Teuchos::RCP<LineVector>*>(latlongdegrees_rcp_view_ptr);

            LineVector latlongdegrees = *(*latlongdegrees_rcp_view);

            Vector neighbors = Spatial::NearestNeighbor<float, ExecutionSpace>(latlongdegrees);

            typedef ViewBuilder<DataTypeKind::Single, 1, ExecutionSpaceKind::OpenMP>::ViewType view_type;
            view_type*                                                                         neighbors_view_ptr = new view_type("neighbors", neighbors.extent(0));

            for(size_type i0 = 0; i0 < neighbors.extent(0); ++i0)
            {
                (*neighbors_view_ptr)(i0) = neighbors(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(neighbors_view_ptr);
        }
        case ExecutionSpaceKind::Cuda:
        {
            using ExecutionSpace = Kokkos::Cuda;
            using LineVector     = Kokkos::View<float***, ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<float*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<LineVector>* latlongdegrees_rcp_view = reinterpret_cast<Teuchos::RCP<LineVector>*>(latlongdegrees_rcp_view_ptr);

            LineVector latlongdegrees = *(*latlongdegrees_rcp_view);

            Vector neighbors = Spatial::NearestNeighbor<float, ExecutionSpace>(latlongdegrees);

            typedef ViewBuilder<DataTypeKind::Single, 1, ExecutionSpaceKind::Cuda>::ViewType view_type;
            view_type*                                                                       neighbors_view_ptr = new view_type("neighbors", neighbors.extent(0));

            for(size_type i0 = 0; i0 < neighbors.extent(0); ++i0)
            {
                (*neighbors_view_ptr)(i0) = neighbors(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(neighbors_view_ptr);
        }
        default:
        {
            std::cout << "ShepardNdSingle: Unknown execution space type." << std::endl;
            return nullptr;
        }
    }
}

void* NearestNeighborDouble(void* latlongdegrees_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            using ExecutionSpace = Kokkos::Serial;
            using LineVector     = Kokkos::View<double***, ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<double*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<LineVector>* latlongdegrees_rcp_view = reinterpret_cast<Teuchos::RCP<LineVector>*>(latlongdegrees_rcp_view_ptr);

            LineVector latlongdegrees = *(*latlongdegrees_rcp_view);

            Vector neighbors = Spatial::NearestNeighbor<double, ExecutionSpace>(latlongdegrees);

            typedef ViewBuilder<DataTypeKind::Double, 1, ExecutionSpaceKind::Serial>::ViewType view_type;
            view_type*                                                                         neighbors_view_ptr = new view_type("neighbors", neighbors.extent(0));

            for(size_type i0 = 0; i0 < neighbors.extent(0); ++i0)
            {
                (*neighbors_view_ptr)(i0) = neighbors(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(neighbors_view_ptr);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            using ExecutionSpace = Kokkos::OpenMP;
            using LineVector     = Kokkos::View<double***, ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<double*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<LineVector>* latlongdegrees_rcp_view = reinterpret_cast<Teuchos::RCP<LineVector>*>(latlongdegrees_rcp_view_ptr);

            LineVector latlongdegrees = *(*latlongdegrees_rcp_view);

            Vector neighbors = Spatial::NearestNeighbor<double, ExecutionSpace>(latlongdegrees);

            typedef ViewBuilder<DataTypeKind::Double, 1, ExecutionSpaceKind::OpenMP>::ViewType view_type;
            view_type*                                                                         neighbors_view_ptr = new view_type("neighbors", neighbors.extent(0));

            for(size_type i0 = 0; i0 < neighbors.extent(0); ++i0)
            {
                (*neighbors_view_ptr)(i0) = neighbors(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(neighbors_view_ptr);
        }
        case ExecutionSpaceKind::Cuda:
        {
            using ExecutionSpace = Kokkos::Cuda;
            using LineVector     = Kokkos::View<double***, ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<double*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<LineVector>* latlongdegrees_rcp_view = reinterpret_cast<Teuchos::RCP<LineVector>*>(latlongdegrees_rcp_view_ptr);

            LineVector latlongdegrees = *(*latlongdegrees_rcp_view);

            Vector neighbors = Spatial::NearestNeighbor<double, ExecutionSpace>(latlongdegrees);

            typedef ViewBuilder<DataTypeKind::Double, 1, ExecutionSpaceKind::Cuda>::ViewType view_type;
            view_type*                                                                       neighbors_view_ptr = new view_type("neighbors", neighbors.extent(0));

            for(size_type i0 = 0; i0 < neighbors.extent(0); ++i0)
            {
                (*neighbors_view_ptr)(i0) = neighbors(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(neighbors_view_ptr);
        }
        default:
        {
            std::cout << "ShepardNdSingle: Unknown execution space type." << std::endl;
            return nullptr;
        }
    }
}
