
#pragma once

#ifndef TEMPLATEMANAGER
#    include "TemplateManager.hpp"
#endif

namespace KokkosDotNET
{
    template<typename TypeSeq, typename BaseT, typename ObjectT>
    TemplateManager<TypeSeq, BaseT, ObjectT>::TemplateManager()
    {
        int sz = Sacado::mpl::size<TypeSeq>::value;
        objects.resize(sz);
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    TemplateManager<TypeSeq, BaseT, ObjectT>::~TemplateManager()
    {
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    template<typename BuilderOpT>
    void TemplateManager<TypeSeq, BaseT, ObjectT>::buildObjects(const BuilderOpT& builder)
    {
        Sacado::mpl::for_each_no_kokkos<TypeSeq>(BuildObject<BuilderOpT>(objects, builder));
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    void TemplateManager<TypeSeq, BaseT, ObjectT>::buildObjects()
    {
        DefaultBuilderOp builder;
        (*this).template buildObjects<DefaultBuilderOp>(builder);
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    template<typename ScalarT>
    Teuchos::RCP<BaseT> TemplateManager<TypeSeq, BaseT, ObjectT>::getAsBase()
    {
        int idx = Sacado::mpl::find<TypeSeq, ScalarT>::value;
        return objects[idx];
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    template<typename ScalarT>
    Teuchos::RCP<const BaseT> TemplateManager<TypeSeq, BaseT, ObjectT>::getAsBase() const
    {
        int idx = Sacado::mpl::find<TypeSeq, ScalarT>::value;
        return objects[idx];
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    template<typename ScalarT>
    Teuchos::RCP<typename Sacado::mpl::apply<ObjectT, ScalarT>::type> TemplateManager<TypeSeq, BaseT, ObjectT>::getAsObject()
    {
        int idx = Sacado::mpl::find<TypeSeq, ScalarT>::value;
        return Teuchos::rcp_dynamic_cast<typename Sacado::mpl::apply<ObjectT, ScalarT>::type>(objects[idx], true);
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    template<typename ScalarT>
    Teuchos::RCP<const typename Sacado::mpl::apply<ObjectT, ScalarT>::type> TemplateManager<TypeSeq, BaseT, ObjectT>::getAsObject() const
    {
        int idx = Sacado::mpl::find<TypeSeq, ScalarT>::value;
        return Teuchos::rcp_dynamic_cast<const typename Sacado::mpl::apply<ObjectT, ScalarT>::type>(objects[idx], true);
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    typename TemplateManager<TypeSeq, BaseT, ObjectT>::iterator TemplateManager<TypeSeq, BaseT, ObjectT>::begin()
    {
        return KokkosDotNET::TemplateIterator<TypeSeq, BaseT, ObjectT>(*this, objects.begin());
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    typename TemplateManager<TypeSeq, BaseT, ObjectT>::const_iterator TemplateManager<TypeSeq, BaseT, ObjectT>::begin() const
    {
        return KokkosDotNET::ConstTemplateIterator<TypeSeq, BaseT, ObjectT>(*this, objects.begin());
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    typename TemplateManager<TypeSeq, BaseT, ObjectT>::iterator TemplateManager<TypeSeq, BaseT, ObjectT>::end()
    {
        return KokkosDotNET::TemplateIterator<TypeSeq, BaseT, ObjectT>(*this, objects.end());
    }

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    typename TemplateManager<TypeSeq, BaseT, ObjectT>::const_iterator TemplateManager<TypeSeq, BaseT, ObjectT>::end() const
    {
        return KokkosDotNET::ConstTemplateIterator<TypeSeq, BaseT, ObjectT>(*this, objects.end());
    }
}
