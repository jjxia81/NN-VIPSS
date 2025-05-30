// =========================================================
//
//	Methods for Vector2TC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R2VectorTC<Coord>::R2VectorTC( const Coord& _du, const Coord& _dv )
{
    du = _du; dv = _dv;
}

// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R2VectorTC<Coord>&
R2VectorTC<Coord>::operator += ( const R2VectorTC<Coord>& v )
{
    du += v.du;
    dv += v.dv;
    return *this;
}

template<class Coord>
inline R2VectorTC<Coord>&
R2VectorTC<Coord>::operator -= ( const R2VectorTC<Coord>& v )
{
    du -= v.du;
    dv -= v.dv;
    return *this;
}

template<class Coord>
inline R2VectorTC<Coord>&
R2VectorTC<Coord>::operator *= ( const Coord& s )
{
    du *= s;
    dv *= s;
    return *this;
}

template<class Coord>
inline R2VectorTC<Coord>&
R2VectorTC<Coord>::operator /= ( const Coord& s )
{
    du /= s;
    dv /= s;
    return *this;
}

// -------------------------------------
// unary operators
// -------------------------------------

template<class Coord>
inline R2VectorTC<Coord>
R2VectorTC<Coord>::operator + () const
{
    return *this;
}

template<class Coord>
inline R2VectorTC<Coord>
R2VectorTC<Coord>::operator - () const
{
    return R2VectorTC<Coord>( -du, -dv );
}


// -------------------------------------
// Self-editing functions
// -------------------------------------

template<class Coord>
inline Coord R2VectorTC<Coord>::Normalize()
{
    const Coord dLen = sqrt(du * du + dv * dv);
    if ( ! RNIsZero( dLen ) ) {
        du = du / dLen;
        dv = dv / dLen;
    }
    return dLen;
}



// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R2VectorTC<Coord>::Write(std::ofstream &out) const
{
    out << du << " " << dv << " ";
}

template<class Coord>
inline WINbool R2VectorTC<Coord>::Read(std::ifstream &in)
{
    in >> du >> dv;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R2VectorTC<Coord>::WriteBinary(std::ofstream &out) const
{
    out.write( (const char *) &du, Dim() * sizeof(Coord) );
}


template<class Coord>
inline WINbool R2VectorTC<Coord>::ReadBinary(std::ifstream &in)
{
    in.read( (char *) &du, Dim() * sizeof(Coord) );
    
    return in.good() ? TRUE : FALSE;
}


template<class Coord>
inline void R2VectorTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    std::cout << du << " " << dv;
    if ( in_bDoReturn == TRUE )
        std::cout << "\n";
    else
        std::cout << " ";
}

