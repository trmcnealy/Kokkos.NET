#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

namespace Geometry
{

    template<typename BoxType>
    struct ObjectBoundingBoxHierarchy
    {
        int     right_child_offset;
        BoxType m_box;

        KOKKOS_FORCEINLINE_FUNCTION ObjectBoundingBoxHierarchy(const int right_child_offset, const BoxType& m_box) : right_child_offset(right_child_offset), m_box(m_box) {}

        ~ObjectBoundingBoxHierarchy() {}

        /**
         * Create a hierarchy from a list of object bounding boxes.  The top level of the hierarchy will contain all boxes.  In the next level
         *  each leave of the tree will contain approximiatly half of the boxes.  The hierarchy continues down until the tree node contains only
         *  a single box
         */
        KOKKOS_FORCEINLINE_FUNCTION void set_right_child_offset(const int right_child_offset_)
        {
            right_child_offset = right_child_offset_;
        }
        KOKKOS_FORCEINLINE_FUNCTION const BoxType& GetBox() const
        {
            return m_box;
        }
        KOKKOS_FORCEINLINE_FUNCTION BoxType& GetBox()
        {
            return m_box;
        }

        //
        //  Right child offset stores one of two things.
        //    If the offset is <= 0 the current object is a terminal node of the tree.  The value is the negative of the
        //  the object number associated with the object represented by the terminal node.  If the tree is created from
        //  inputBoxes that are not std::search::OblectBoundingBoxes, then the object number is the offset into the
        //  vector of inputBoxes.
        //    If the value of right_child_offset is positive it is the offset from the current object to the objects
        //  right child.  Note that the offset to the left child is always one.  Thus for a given object the left child
        //  can be found at this[1] and the right child at this[right_child_offset]
        //
    };

}
