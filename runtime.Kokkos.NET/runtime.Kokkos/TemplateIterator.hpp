
#pragma once

#include <iterator>
#include <vector>

#include "TemplateManager.hpp"

namespace KokkosDotNET
{
    template<typename TypeSeq, typename BaseT, typename ObjectT>
    derived_class TemplateIterator : public std::iterator<std::input_iterator_tag, BaseT>
    {
        TemplateManager<TypeSeq, BaseT, ObjectT>* manager;

        typename std::vector<Teuchos::RCP<BaseT>>::iterator object_iterator;

    public:
        TemplateIterator(TemplateManager<TypeSeq, BaseT, ObjectT>& m, typename std::vector<Teuchos::RCP<BaseT>>::iterator p) : manager(&m), object_iterator(p) {}

        bool operator==(const TemplateIterator& t) const
        {
            return object_iterator == t.objectIterator;
        }

        bool operator!=(const TemplateIterator& t) const
        {
            return object_iterator != t.object_iterator;
        }

        typename TemplateIterator<TypeSeq, BaseT, ObjectT>::reference operator*() const
        {
            return *(*object_iterator);
        }

        typename TemplateIterator<TypeSeq, BaseT, ObjectT>::pointer operator->() const
        {
            return &(*(*object_iterator));
        }

        TemplateIterator& operator++()
        {
            ++object_iterator;
            return *this;
        }

        TemplateIterator operator++(int)
        {
            TemplateIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        Teuchos::RCP<BaseT> rcp() const
        {
            return *object_iterator;
        }
    };

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    derived_class ConstTemplateIterator : public std::iterator<std::input_iterator_tag, BaseT>
    {
        const TemplateManager<TypeSeq, BaseT, ObjectT>* manager;

        typename std::vector<Teuchos::RCP<BaseT>>::const_iterator object_iterator;

    public:
        ConstTemplateIterator(const TemplateManager<TypeSeq, BaseT, ObjectT>& m, typename std::vector<Teuchos::RCP<BaseT>>::const_iterator p) : manager(&m), object_iterator(p)
        {
        }

        bool operator==(const ConstTemplateIterator& t) const
        {
            return object_iterator == t.objectIterator;
        }

        bool operator!=(const ConstTemplateIterator& t) const
        {
            return object_iterator != t.object_iterator;
        }

        const typename ConstTemplateIterator<TypeSeq, BaseT, ObjectT>::reference operator*() const
        {
            return *(*object_iterator);
        }

        const typename ConstTemplateIterator<TypeSeq, BaseT, ObjectT>::pointer operator->() const
        {
            return &(*(*object_iterator));
        }

        ConstTemplateIterator& operator++()
        {
            ++object_iterator;
            return *this;
        }

        ConstTemplateIterator operator++(int)
        {
            ConstTemplateIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        Teuchos::RCP<BaseT> rcp() const
        {
            return *object_iterator;
        }
    };
}
