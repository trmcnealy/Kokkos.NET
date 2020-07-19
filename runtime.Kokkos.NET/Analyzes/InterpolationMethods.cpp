
#include "Analyzes/InterpolationMethods.hpp"

void* Shepard2dSingle(void* xd_rcp_view_ptr, void* zd_rcp_view_ptr, const float& p, void* xi_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            using ExecutionSpace = Kokkos::Serial;
            using PointVector    = Kokkos::View<float* [2], ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<float*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<PointVector>* xd_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xd_rcp_view_ptr);
            Teuchos::RCP<Vector>*      zd_rcp_view = reinterpret_cast<Teuchos::RCP<Vector>*>(zd_rcp_view_ptr);
            Teuchos::RCP<PointVector>* xi_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xi_rcp_view_ptr);

            PointVector xd = *(*xd_rcp_view);
            PointVector xi = *(*xi_rcp_view);
            Vector      zd = *(*zd_rcp_view);

            Vector zi = Interpolation::ShepardNd<float, ExecutionSpace, 2>(xd, zd, p, xi);

            typedef ViewBuilder<DataTypeKind::Single, 1, ExecutionSpaceKind::Serial>::ViewType view_type;
            view_type*                                                                         zi_view_ptr = new view_type("zi", zi.extent(0));

            for(size_type i0 = 0; i0 < zi.extent(0); ++i0)
            {
                (*zi_view_ptr)(i0) = zi(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(zi_view_ptr);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            using ExecutionSpace = Kokkos::OpenMP;
            using PointVector    = Kokkos::View<float* [2], ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<float*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<PointVector>* xd_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xd_rcp_view_ptr);
            Teuchos::RCP<Vector>*      zd_rcp_view = reinterpret_cast<Teuchos::RCP<Vector>*>(zd_rcp_view_ptr);
            Teuchos::RCP<PointVector>* xi_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xi_rcp_view_ptr);

            PointVector xd = *(*xd_rcp_view);
            PointVector xi = *(*xi_rcp_view);
            Vector      zd = *(*zd_rcp_view);

            Vector zi = Interpolation::ShepardNd<float, ExecutionSpace, 2>(xd, zd, p, xi);

            typedef ViewBuilder<DataTypeKind::Single, 1, ExecutionSpaceKind::OpenMP>::ViewType view_type;
            view_type*                                                                         zi_view_ptr = new view_type("zi", zi.extent(0));

            for(size_type i0 = 0; i0 < zi.extent(0); ++i0)
            {
                (*zi_view_ptr)(i0) = zi(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(zi_view_ptr);
        }
        case ExecutionSpaceKind::Cuda:
        {
            using ExecutionSpace = Kokkos::Cuda;
            using PointVector    = Kokkos::View<float* [2], ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<float*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<PointVector>* xd_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xd_rcp_view_ptr);
            Teuchos::RCP<Vector>*      zd_rcp_view = reinterpret_cast<Teuchos::RCP<Vector>*>(zd_rcp_view_ptr);
            Teuchos::RCP<PointVector>* xi_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xi_rcp_view_ptr);

            PointVector xd = *(*xd_rcp_view);
            PointVector xi = *(*xi_rcp_view);
            Vector      zd = *(*zd_rcp_view);

            Vector zi = Interpolation::ShepardNd<float, ExecutionSpace, 2>(xd, zd, p, xi);

            typedef ViewBuilder<DataTypeKind::Single, 1, ExecutionSpaceKind::Cuda>::ViewType view_type;
            view_type*                                                                       zi_view_ptr = new view_type("zi", zi.extent(0));

            for(size_type i0 = 0; i0 < zi.extent(0); ++i0)
            {
                (*zi_view_ptr)(i0) = zi(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(zi_view_ptr);
        }
        default:
        {
            std::cout << "ShepardNdSingle: Unknown execution space type." << std::endl;
            return nullptr;
        }
    }
}

void* Shepard2dDouble(void* xd_rcp_view_ptr, void* zd_rcp_view_ptr, const double& p, void* xi_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            using ExecutionSpace = Kokkos::Serial;
            using PointVector    = Kokkos::View<double* [2], ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<double*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<PointVector>* xd_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xd_rcp_view_ptr);
            Teuchos::RCP<Vector>*      zd_rcp_view = reinterpret_cast<Teuchos::RCP<Vector>*>(zd_rcp_view_ptr);
            Teuchos::RCP<PointVector>* xi_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xi_rcp_view_ptr);

            PointVector xd = *(*xd_rcp_view);
            PointVector xi = *(*xi_rcp_view);
            Vector      zd = *(*zd_rcp_view);

            Vector zi = Interpolation::ShepardNd<double, ExecutionSpace, 2>(xd, zd, p, xi);

            typedef ViewBuilder<DataTypeKind::Double, 1, ExecutionSpaceKind::Serial>::ViewType view_type;
            view_type*                                                                         zi_view_ptr = new view_type("zi", zi.extent(0));

            for(size_type i0 = 0; i0 < zi.extent(0); ++i0)
            {
                (*zi_view_ptr)(i0) = zi(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(zi_view_ptr);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            using ExecutionSpace = Kokkos::OpenMP;
            using PointVector    = Kokkos::View<double* [2], ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<double*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<PointVector>* xd_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xd_rcp_view_ptr);
            Teuchos::RCP<Vector>*      zd_rcp_view = reinterpret_cast<Teuchos::RCP<Vector>*>(zd_rcp_view_ptr);
            Teuchos::RCP<PointVector>* xi_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xi_rcp_view_ptr);

            PointVector xd = *(*xd_rcp_view);
            PointVector xi = *(*xi_rcp_view);
            Vector      zd = *(*zd_rcp_view);

            Vector zi = Interpolation::ShepardNd<double, ExecutionSpace, 2>(xd, zd, p, xi);

            typedef ViewBuilder<DataTypeKind::Double, 1, ExecutionSpaceKind::OpenMP>::ViewType view_type;
            view_type*                                                                         zi_view_ptr = new view_type("zi", zi.extent(0));

            for(size_type i0 = 0; i0 < zi.extent(0); ++i0)
            {
                (*zi_view_ptr)(i0) = zi(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(zi_view_ptr);
        }
        case ExecutionSpaceKind::Cuda:
        {
            using ExecutionSpace = Kokkos::Cuda;
            using PointVector    = Kokkos::View<double* [2], ExecutionSpace::array_layout, ExecutionSpace>;
            using Vector         = Kokkos::View<double*, ExecutionSpace::array_layout, ExecutionSpace>;

            Teuchos::RCP<PointVector>* xd_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xd_rcp_view_ptr);
            Teuchos::RCP<Vector>*      zd_rcp_view = reinterpret_cast<Teuchos::RCP<Vector>*>(zd_rcp_view_ptr);
            Teuchos::RCP<PointVector>* xi_rcp_view = reinterpret_cast<Teuchos::RCP<PointVector>*>(xi_rcp_view_ptr);

            PointVector xd = *(*xd_rcp_view);
            PointVector xi = *(*xi_rcp_view);
            Vector      zd = *(*zd_rcp_view);

            Vector zi = Interpolation::ShepardNd<double, ExecutionSpace, 2>(xd, zd, p, xi);

            typedef ViewBuilder<DataTypeKind::Double, 1, ExecutionSpaceKind::Cuda>::ViewType view_type;
            view_type*                                                                       zi_view_ptr = new view_type("zi", zi.extent(0));

            for(size_type i0 = 0; i0 < zi.extent(0); ++i0)
            {
                (*zi_view_ptr)(i0) = zi(i0);
            }

            return (void*)new Teuchos::RCP<view_type>(zi_view_ptr);
        }
        default:
        {
            std::cout << "ShepardNdSingle: Unknown execution space type." << std::endl;
            return nullptr;
        }
    }
}
