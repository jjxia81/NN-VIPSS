// =========================================================
//
//	Methods for R3CoVectorTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------


template<class Coord>
inline
R3CoVectorTC<Coord>::R3CoVectorTC( const Coord& _dx, const Coord& _dy, const Coord& _dz )
{
    dx = _dx; dy = _dy; dz = _dz;
}

// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R3CoVectorTC<Coord>&
R3CoVectorTC<Coord>::operator += ( const R3CoVectorTC<Coord>& v)
{
    dx += v.dx;
    dy += v.dy;
    dz += v.dz;
    return *this;
}

template<class Coord>
inline R3CoVectorTC<Coord>&
R3CoVectorTC<Coord>::operator -= ( const R3CoVectorTC<Coord>& v )
{
    dx -= v.dx;
    dy -= v.dy;
    dz -= v.dz;
    return *this;
}

template<class Coord>
inline R3CoVectorTC<Coord>&
R3CoVectorTC<Coord>::operator *= ( const Coord& s )
{
    dx *= s;
    dy *= s;
    dz *= s;
    return *this;
}

template<class Coord>
inline R3CoVectorTC<Coord>&
R3CoVectorTC<Coord>::operator /= ( const Coord& s )
{
    Coord s1 = Coord(1.0f) / s;
    dx *= s1;
    dy *= s1;
    dz *= s1;
    return *this;
}

// -------------------------------------
// unary operators
// -------------------------------------

template<class Coord>
inline R3CoVectorTC<Coord>
R3CoVectorTC<Coord>::operator + () const
{
    return *this;
}

template<class Coord>
inline R3CoVectorTC<Coord>
R3CoVectorTC<Coord>::operator - () const
{
    return R3CoVectorTC<Coord>( -dx, -dy, -dz );
}



// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R3CoVectorTC<Coord>::Write(std::ofstream &out) const
{
    out << dx << " " << dy << " " << dz << " ";
}


template<class Coord>
inline void R3CoVectorTC<Coord>::WriteBinary(std::ofstream &out) const
{
    out.write( (const char *) &dx, Dim() * sizeof(Coord) );
}

template<class Coord>
inline WINbool R3CoVectorTC<Coord>::Read(std::ifstream &in)
{
    in >> dx >> dy >> dz;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline WINbool R3CoVectorTC<Coord>::ReadBinary(std::ifstream &in)
{
    in.read( (char *) &dx, Dim() * sizeof(Coord) );

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R3CoVectorTC<Coord>::Print( WINbool in_bDoReturn ) const
{
    std::cout << dx << " " << dy << " " << dz;
    if ( in_bDoReturn == TRUE )
        std::cout << "\n";
    else
        std::cout << " ";
}
