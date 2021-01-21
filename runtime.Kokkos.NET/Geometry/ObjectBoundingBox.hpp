#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

#include <Geometry/BoundingBox.hpp>

namespace Geometry
{
    template<typename BoxType>
    struct ObjectBoundingBox
    {
        typedef typename BoxType::value_type coordinate_t;

    private:
        BoxType m_box;
        int     obj_num;

    public:
        KOKKOS_FORCEINLINE_FUNCTION ObjectBoundingBox() :
            ///< Default empty box and invalid object id
            obj_num(-1)
        {
        }
        KOKKOS_FORCEINLINE_FUNCTION ObjectBoundingBox(const BoxType& box, ///< Explicitly defined constructor
                                                      const int      obj_num_) :
            m_box(box),
            obj_num(obj_num_)
        {
        }
        KOKKOS_FORCEINLINE_FUNCTION ObjectBoundingBox(const ObjectBoundingBox& box) : m_box(box.GetBox()), obj_num(box.obj_num) {}
        KOKKOS_FORCEINLINE_FUNCTION ObjectBoundingBox(ObjectBoundingBox&&) = default; ///< Default Move constructor
        KOKKOS_FORCEINLINE_FUNCTION ~ObjectBoundingBox() {}                           ///< Destructor

        KOKKOS_FORCEINLINE_FUNCTION ObjectBoundingBox& operator=(const ObjectBoundingBox&)
        {
            obj_num = P.get_object_number();
            m_box   = P.GetBox();
            return *this;
        }
        KOKKOS_FORCEINLINE_FUNCTION ObjectBoundingBox& operator=(ObjectBoundingBox&&) = default; ///< Default move assignment
        ///
        ///  Explicity set or extract the object index for this bounding box
        ///
        KOKKOS_FORCEINLINE_FUNCTION void set_object_number(const int& obj_num_)
        {
            obj_num = obj_num_;
        }
        KOKKOS_FORCEINLINE_FUNCTION int get_object_number() const
        {
            return obj_num;
        }
        KOKKOS_FORCEINLINE_FUNCTION const BoxType& GetBox() const
        {
            return m_box;
        }
        KOKKOS_FORCEINLINE_FUNCTION void SetBox(const BoxType& box)
        {
            m_box.set_box(box.get_x_min(), box.get_y_min(), box.get_z_min(), box.get_x_max(), box.get_y_max(), box.get_z_max());
        }
        KOKKOS_FORCEINLINE_FUNCTION void AddBox(const BoxType& box)
        {
            Geometry::add_to_box(m_box, box);
        }
        KOKKOS_FORCEINLINE_FUNCTION void Reset()
        {
            m_box = BoxType();
        }
        KOKKOS_FORCEINLINE_FUNCTION BoxType& GetBox()
        {
            return m_box;
        }
    };

    template<typename BoxType>
    std::ostream& operator<<(std::ostream& output, const ObjectBoundingBox<BoxType>& box)
    {
        output << "Min corner " << box.GetBox().get_x_min() << " " << box.GetBox().get_y_min() << " " << box.GetBox().get_z_min() << std::endl;
        output << "Max corner " << box.GetBox().get_x_max() << " " << box.GetBox().get_y_max() << " " << box.GetBox().get_z_max() << std::endl;
        output << "object number " << box.get_object_number() << std::endl;
        return output;
    }
}
