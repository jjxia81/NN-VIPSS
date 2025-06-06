// =========================================================
//
//	Methods for R4CoVectorTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R4CoVectorTC<Coord>::R4CoVectorTC( const Coord& _dx, const Coord& _dy, const Coord& _dz, const Coord & _dw )
{
    dx = _dx; dy = _dy; dz = _dz; dw = _dw;
}

// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R4CoVectorTC<Coord>&
R4CoVectorTC<Coord>::operator += ( const R4CoVectorTC<Coord>& v)
{
    dx += v[0];
    dy += v[1];
    dz += v[2];
    dw += v[3];
    return *this;
}

template<class Coord>
inline R4CoVectorTC<Coord>&
R4CoVectorTC<Coord>::operator -= ( const R4CoVectorTC<Coord>& v )
{
    dx -= v[0];
    dy -= v[1];
    dz -= v[2];
    dw -= v[3];
    return *this;
}

template<class Coord>
inline R4CoVectorTC<Coord>&
R4CoVectorTC<Coord>::operator *= ( const Coord& s )
{
    dx *= s;
    dy *= s;
    dz *= s;
    dw *= s;
    return *this;
}

template<class Coord>
inline R4CoVectorTC<Coord>&
R4CoVectorTC<Coord>::operator /= ( const Coord& s )
{
    ASSERT( !RNIsZero( s ) );
    Coord s1 = Coord(1.0f) / s;
    dx *= s1;
    dy *= s1;
    dz *= s1;
    dw *= s1;
    return *this;
}

// -------------------------------------
// unary operators
// -------------------------------------

template<class Coord>
inline R4CoVectorTC<Coord>
R4CoVectorTC<Coord>::operator + () const
{
    return *this;
}

template<class Coord>
inline R4CoVectorTC<Coord>
R4CoVectorTC<Coord>::operator - () const
{
    return R4CoVectorTC<Coord>( -dx, -dy, -dz, -dw );
}



// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R4CoVectorTC<Coord>::Write(std::ofstream &out) const
{
    out << dx << " " << dy << " " << dz << " " << dw << " ";
}

template<class Coord>
inline WINbool R4CoVectorTC<Coord>::Read(std::ifstream &in)
{
    in >> dx >> dy >> dz >> dw;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R4CoVectorTC<Coord>::Print( WINbool in_bDoReturn ) const
{
    std::cout << dx << " " << dy << " " << dz << " " << dw;
    if ( in_bDoReturn == TRUE )
        std::cout << "\n";
    else
        std::cout << " ";
}
