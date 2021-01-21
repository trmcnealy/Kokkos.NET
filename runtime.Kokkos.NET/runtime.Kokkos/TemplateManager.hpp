
#pragma once

#include <typeinfo>
#include <vector>

#include <Teuchos_RCP.hpp>

#include <Sacado_mpl_size.hpp>
#include <Sacado_mpl_find.hpp>
#include <Sacado_mpl_for_each.hpp>
#include <Sacado_mpl_apply.hpp>

#include "ViewTypes.hpp"

namespace KokkosDotNET
{
    template<typename TypeSeq, typename BaseT, typename ObjectT>
    class TemplateIterator;

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    class ConstTemplateIterator;

    template<typename TypeSeq, typename BaseT, typename ObjectT>
    class TemplateManager
    {
        std::vector<Teuchos::RCP<BaseT>> objects;

        struct type_info_less
        {
            bool operator()(const std::type_info* a, const std::type_info* b)
            {
                return a->before(*b);
            }
        };

        template<typename BuilderOpT>
        struct BuildObject
        {
            std::vector<Teuchos::RCP<BaseT>>& objects;
            const BuilderOpT&                 builder;

            BuildObject(std::vector<Teuchos::RCP<BaseT>>& objects_, const BuilderOpT& builder_) : objects(objects_), builder(builder_) {}

            template<typename T>
            void operator()(const std::string& label) const
            {
                int idx      = Sacado::mpl::find<TypeSeq, T>::value;
                objects[idx] = builder.template build<T>(label);
            }

            template<typename T>
            void operator()(const std::string& label, const size_type& n0) const
            {
                int idx      = Sacado::mpl::find<TypeSeq, T>::value;
                objects[idx] = builder.template build<T>(label, n0);
            }

            template<typename T>
            void operator()(const std::string& label, const size_type& n0, const size_type& n1) const
            {
                int idx      = Sacado::mpl::find<TypeSeq, T>::value;
                objects[idx] = builder.template build<T>(label, n0, n1);
            }

            template<typename T>
            void operator()(const std::string& label, const size_type& n0, const size_type& n1, const size_type& n2) const
            {
                int idx      = Sacado::mpl::find<TypeSeq, T>::value;
                objects[idx] = builder.template build<T>(label, n0, n1, n2);
            }
        };

        friend class TemplateIterator<TypeSeq, BaseT, ObjectT>;

    public:
        typedef TemplateIterator<TypeSeq, BaseT, ObjectT> iterator;

        typedef ConstTemplateIterator<TypeSeq, BaseT, ObjectT> const_iterator;

        struct DefaultBuilderOp
        {
            template<class ScalarT>
            Teuchos::RCP<BaseT> build() const
            {
                typedef typename Sacado::mpl::apply<ObjectT, ScalarT>::type type;
                return Teuchos::rcp(dynamic_cast<BaseT*>(new type));
            }
        };

        TemplateManager();

        ~TemplateManager();

        template<typename BuilderOpT>
        void buildObjects(const BuilderOpT& builder);

        void buildObjects();

        template<typename ScalarT>
        Teuchos::RCP<BaseT> getAsBase();

        template<typename ScalarT>
        Teuchos::RCP<const BaseT> getAsBase() const;

        template<typename ScalarT>
        Teuchos::RCP<typename Sacado::mpl::apply<ObjectT, ScalarT>::type> getAsObject();

        template<typename ScalarT>
        Teuchos::RCP<const typename Sacado::mpl::apply<ObjectT, ScalarT>::type> getAsObject() const;

        iterator begin();

        const_iterator begin() const;

        iterator end();

        const_iterator end() const;
    };
}

#define TEMPLATEMANAGER

#include "TemplateManager_Def.hpp"

#undef TEMPLATEMANAGER

#include "TemplateIterator.hpp"
