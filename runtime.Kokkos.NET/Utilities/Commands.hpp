
#include <functional> //  std::function, std::bind
#include <sqlite3.h>
#include <string> //  std::string
#include <sstream> //  std::stringstream
#include <utility> //  std::move
#include <system_error> //  std::system_error, std::error_code, std::make_error_code
#include <vector> //  std::vector
#include <memory> //  std::make_shared, std::shared_ptr
#include <map> //  std::map
#include <type_traits> //  std::decay, std::is_same
#include <algorithm> //  std::iter_swap

// #include "pragma.h"

#include <string> //  std::string
#include <sqlite3.h>
#include <functional> //  std::function
#include <memory> // std::shared_ptr



namespace sqlite_orm
{
    namespace internal
    {
        struct storage_base;
    }

    struct pragma_t
    {
        using get_connection_t = std::function<internal::connection_ref()>;

        pragma_t(get_connection_t get_connection_) : get_connection(std::move(get_connection_)) {}

        sqlite_orm::journal_mode journal_mode() { return this->get_pragma<sqlite_orm::journal_mode>("journal_mode"); }

        void journal_mode(sqlite_orm::journal_mode value)
        {
            this->_journal_mode = -1;
            this->set_pragma("journal_mode", value);
            this->_journal_mode = static_cast<decltype(this->_journal_mode)>(value);
        }

        int synchronous() { return this->get_pragma<int>("synchronous"); }

        void synchronous(int value)
        {
            this->_synchronous = -1;
            this->set_pragma("synchronous", value);
            this->_synchronous = value;
        }

        int user_version() { return this->get_pragma<int>("user_version"); }

        void user_version(int value) { this->set_pragma("user_version", value); }

        int auto_vacuum() { return this->get_pragma<int>("auto_vacuum"); }

        void auto_vacuum(int value) { this->set_pragma("auto_vacuum", value); }

    protected:
        friend struct storage_base;

    public:
        int              _synchronous  = -1;
        signed char      _journal_mode = -1; //  if != -1 stores static_cast<sqlite_orm::journal_mode>(journal_mode)
        get_connection_t get_connection;

        template<class T>
        T get_pragma(const std::string& name)
        {
            auto connection = this->get_connection();
            auto query      = "PRAGMA " + name;
            T    res;
            auto db = connection.get();
            auto rc = sqlite3_exec(
                db,
                query.c_str(),
                [](void* data, int argc, char** argv, char**) -> int {
                    auto& res = *(T*)data;
                    if(argc)
                    {
                        res = row_extractor<T>().extract(argv[0]);
                    }
                    return 0;
                },
                &res,
                nullptr);
            if(rc == SQLITE_OK)
            {
                return res;
            }
            else
            {
                throw std::system_error(std::error_code(sqlite3_errcode(db), get_sqlite_error_category()), sqlite3_errmsg(db));
            }
        }

        /**
         *  Yevgeniy Zakharov: I wanted to refactore this function with statements and value bindings
         *  but it turns out that bindings in pragma statements are not supported.
         */
        template<class T>
        void set_pragma(const std::string& name, const T& value, sqlite3* db = nullptr)
        {
            auto con = this->get_connection();
            if(!db)
            {
                db = con.get();
            }
            std::stringstream ss;
            ss << "PRAGMA " << name << " = " << value;
            auto query = ss.str();
            auto rc    = sqlite3_exec(db, query.c_str(), nullptr, nullptr, nullptr);
            if(rc != SQLITE_OK)
            {
                throw std::system_error(std::error_code(sqlite3_errcode(db), get_sqlite_error_category()), sqlite3_errmsg(db));
            }
        }

        void set_pragma(const std::string& name, const sqlite_orm::journal_mode& value, sqlite3* db = nullptr)
        {
            auto con = this->get_connection();
            if(!db)
            {
                db = con.get();
            }
            std::stringstream ss;
            ss << "PRAGMA " << name << " = " << internal::to_string(value);
            auto query = ss.str();
            auto rc    = sqlite3_exec(db, query.c_str(), nullptr, nullptr, nullptr);
            if(rc != SQLITE_OK)
            {
                throw std::system_error(std::error_code(sqlite3_errcode(db), get_sqlite_error_category()), sqlite3_errmsg(db));
            }
        }
    };
}

// #include "limit_accesor.h"

#include <sqlite3.h>
#include <map> 
#include <functional> 
#include <memory> 

namespace sqlite_orm
{
    namespace internal
    {
        struct limit_accesor
        {
            using get_connection_t = std::function<connection_ref()>;

            limit_accesor(get_connection_t get_connection_) : get_connection(std::move(get_connection_)) {}

            int length() { return this->get(SQLITE_LIMIT_LENGTH); }

            void length(int newValue) { this->set(SQLITE_LIMIT_LENGTH, newValue); }

            int sql_length() { return this->get(SQLITE_LIMIT_SQL_LENGTH); }

            void sql_length(int newValue) { this->set(SQLITE_LIMIT_SQL_LENGTH, newValue); }

            int column() { return this->get(SQLITE_LIMIT_COLUMN); }

            void column(int newValue) { this->set(SQLITE_LIMIT_COLUMN, newValue); }

            int expr_depth() { return this->get(SQLITE_LIMIT_EXPR_DEPTH); }

            void expr_depth(int newValue) { this->set(SQLITE_LIMIT_EXPR_DEPTH, newValue); }

            int compound_select() { return this->get(SQLITE_LIMIT_COMPOUND_SELECT); }

            void compound_select(int newValue) { this->set(SQLITE_LIMIT_COMPOUND_SELECT, newValue); }

            int vdbe_op() { return this->get(SQLITE_LIMIT_VDBE_OP); }

            void vdbe_op(int newValue) { this->set(SQLITE_LIMIT_VDBE_OP, newValue); }

            int function_arg() { return this->get(SQLITE_LIMIT_FUNCTION_ARG); }

            void function_arg(int newValue) { this->set(SQLITE_LIMIT_FUNCTION_ARG, newValue); }

            int attached() { return this->get(SQLITE_LIMIT_ATTACHED); }

            void attached(int newValue) { this->set(SQLITE_LIMIT_ATTACHED, newValue); }

            int like_pattern_length() { return this->get(SQLITE_LIMIT_LIKE_PATTERN_LENGTH); }

            void like_pattern_length(int newValue) { this->set(SQLITE_LIMIT_LIKE_PATTERN_LENGTH, newValue); }

            int variable_number() { return this->get(SQLITE_LIMIT_VARIABLE_NUMBER); }

            void variable_number(int newValue) { this->set(SQLITE_LIMIT_VARIABLE_NUMBER, newValue); }

            int trigger_depth() { return this->get(SQLITE_LIMIT_TRIGGER_DEPTH); }

            void trigger_depth(int newValue) { this->set(SQLITE_LIMIT_TRIGGER_DEPTH, newValue); }

#if SQLITE_VERSION_NUMBER >= 3008007
            int worker_threads() { return this->get(SQLITE_LIMIT_WORKER_THREADS); }

            void worker_threads(int newValue) { this->set(SQLITE_LIMIT_WORKER_THREADS, newValue); }
#endif

        protected:
            get_connection_t get_connection;

            friend struct storage_base;

            /**
             *  Stores limit set between connections.
             */
            std::map<int, int> limits;

            int get(int id)
            {
                auto connection = this->get_connection();
                return sqlite3_limit(connection.get(), id, -1);
            }

            void set(int id, int newValue)
            {
                this->limits[id] = newValue;
                auto connection  = this->get_connection();
                sqlite3_limit(connection.get(), id, newValue);
            }
        };
    }
}


#include <functional> 

namespace sqlite_orm
{
    namespace internal
    {
        /**
         *  Class used as a guard for a transaction. Calls `ROLLBACK` in destructor.
         *  Has explicit `commit()` and `rollback()` functions. After explicit function is fired
         *  guard won't do anything in d-tor. Also you can set `commit_on_destroy` to true to
         *  make it call `COMMIT` on destroy.
         */
        struct transaction_guard_t
        {
            /**
             *  This is a public lever to tell a guard what it must do in its destructor
             *  if `gotta_fire` is true
             */
            bool commit_on_destroy = false;

            transaction_guard_t(connection_ref connection_, std::function<void()> commit_func_, std::function<void()> rollback_func_) :
                connection(std::move(connection_)), commit_func(std::move(commit_func_)), rollback_func(std::move(rollback_func_))
            {
            }

            ~transaction_guard_t()
            {
                if(this->gotta_fire)
                {
                    if(!this->commit_on_destroy)
                    {
                        this->rollback_func();
                    }
                    else
                    {
                        this->commit_func();
                    }
                }
            }

            /**
             *  Call `COMMIT` explicitly. After this call
             *  guard will not call `COMMIT` or `ROLLBACK`
             *  in its destructor.
             */
            void commit()
            {
                this->commit_func();
                this->gotta_fire = false;
            }

            /**
             *  Call `ROLLBACK` explicitly. After this call
             *  guard will not call `COMMIT` or `ROLLBACK`
             *  in its destructor.
             */
            void rollback()
            {
                this->rollback_func();
                this->gotta_fire = false;
            }

        protected:
            connection_ref        connection;
            std::function<void()> commit_func;
            std::function<void()> rollback_func;
            bool                  gotta_fire = true;
        };
    }
}

