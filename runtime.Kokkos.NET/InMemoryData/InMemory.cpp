

//#include <InMemoryData/storage.h>
//
//#include <InMemoryData/table.h>

#include <Types.hpp>

// using namespace sqlite_orm;
//
// int dll_method(int, char**)
//{
//    struct RapArtist
//    {
//        int         id;
//        std::string name;
//    };
//
//    auto storage = make_storage(":memory:", make_table("rap_artists", make_column("id", &RapArtist::id, primary_key()), make_column("name", &RapArtist::name)));
//
//    return 0;
//}

class CGenLexSource
{
public:
    virtual ~CGenLexSource()          = default;
    virtual wchar_t NextChar()        = 0;
    virtual void    Pushback(wchar_t) = 0;
    virtual void    Reset()           = 0;
};

class CTextLexSource final : public CGenLexSource
{
    const wchar_t* m_pSrcBuf;
    const wchar_t* m_pStart;

public:
    explicit CTextLexSource(const wchar_t* pSrc)
    {
        SetString(pSrc);
    }

    wchar_t NextChar() override
    {
        if (!m_pSrcBuf)
        {
            return 0;
        }
        return *m_pSrcBuf++ ? m_pSrcBuf[-1] : 0;
    }

    void Pushback(wchar_t) override
    {
        if (m_pSrcBuf)
        {
            --m_pSrcBuf;
        }
    }

    void Reset() override { m_pSrcBuf = m_pStart; }
    void SetString(const wchar_t* pSrc) { m_pSrcBuf = m_pStart = pSrc; }
};

struct LexEl
{
    wchar_t cFirst;
    wchar_t cLast;
    uint32  wGotoState;
    uint32  wReturnTok;
    uint32  wInstructions;
};

class CGenLexer
{
    wchar_t*       m_pTokenBuf;
    int            m_nCurrentLine;
    int            m_nCurBufSize;
    CGenLexSource* m_pSrc;
    LexEl*         m_pTable;

public:
    CGenLexer(LexEl* pTbl, CGenLexSource* pSrc);

    ~CGenLexer();

    int NextToken();

    wchar_t* GetTokenText() const { return m_pTokenBuf; }
    int      GetLineNum() const { return m_nCurrentLine; }
    void     Reset();
};
