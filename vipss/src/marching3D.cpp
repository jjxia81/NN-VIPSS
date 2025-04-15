#include "marching3D.h"
#include <array>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <numeric>

namespace {
// lookup table for marching 4-simplices
constexpr std::array<std::array<size_t, 2>, 10> tet_edges4D = {
    {
        {{0, 1}}, {{0, 2}}, {{0, 3}}, {{0, 4}}, {{1, 2}},
        {{1, 3}}, {{1, 4}}, {{2, 3}}, {{2, 4}}, {{3, 4}}
    }
};

const std::vector<std::vector<size_t> > tet_table4D_verts = {
    {}, {3, 6, 8, 9}, {2, 5, 7, 9}, {2, 3, 5, 6, 7, 8}, {1, 4, 7, 8},
    {1, 3, 4, 6, 7, 9}, {1, 2, 4, 5, 8, 9}, {1, 2, 3, 4, 5, 6}, {0, 4, 5, 6},
    {0, 3, 4, 5, 8, 9}, {0, 2, 4, 6, 7, 9}, {0, 2, 3, 4, 7, 8}, {0, 1, 5, 6, 7, 8},
    {0, 1, 3, 5, 7, 9}, {0, 1, 2, 6, 8, 9}, {0, 1, 2, 3}, {0, 1, 2, 3},
    {0, 1, 2, 6, 8, 9}, {0, 1, 3, 5, 7, 9}, {0, 1, 5, 6, 7, 8}, {0, 2, 3, 4, 7, 8},
    {0, 2, 4, 6, 7, 9}, {0, 3, 4, 5, 8, 9}, {0, 4, 5, 6}, {1, 2, 3, 4, 5, 6},
    {1, 2, 4, 5, 8, 9}, {1, 3, 4, 6, 7, 9}, {1, 4, 7, 8}, {2, 3, 5, 6, 7, 8},
    {2, 5, 7, 9}, {3, 6, 8, 9}, {}};

// every 4 indices represent a tetrahedron
const std::vector<std::vector<size_t> > tet_table4D_tets = {
    {}, {3, 6, 8, 9}, {2, 5, 7, 9}, {2, 3, 6, 8, 2, 5, 7, 8, 2, 5, 6, 8},
    {1, 4, 7, 8}, {1, 3, 6, 9, 1, 4, 7, 9, 1, 4, 6, 9}, {1, 2, 5, 9, 1, 4, 8, 9, 1, 4, 5, 9},
    {1, 2, 3, 6, 1, 4, 5, 6, 1, 2, 5, 6}, {0, 4, 5, 6},
    {0, 3, 8, 9, 0, 4, 5, 9, 0, 4, 8, 9}, {0, 2, 7, 9, 0, 4, 6, 9, 0, 4, 7, 9},
    {0, 2, 3, 8, 0, 4, 7, 8, 0, 2, 7, 8}, {0, 1, 7, 8, 0, 5, 6, 8, 0, 5, 7, 8},
    {0, 1, 3, 9, 0, 5, 7, 9, 0, 1, 7, 9}, {0, 1, 2, 9, 0, 6, 8, 9, 0, 1, 8, 9},
    {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 9, 0, 6, 8, 9, 0, 1, 8, 9},
    {0, 1, 3, 9, 0, 5, 7, 9, 0, 1, 7, 9}, {0, 1, 7, 8, 0, 5, 6, 8, 0, 5, 7, 8},
    {0, 2, 3, 8, 0, 4, 7, 8, 0, 2, 7, 8}, {0, 2, 7, 9, 0, 4, 6, 9, 0, 4, 7, 9},
    {0, 3, 8, 9, 0, 4, 5, 9, 0, 4, 8, 9}, {0, 4, 5, 6}, {1, 2, 3, 6, 1, 4, 5, 6, 1, 2, 5, 6},
    {1, 2, 5, 9, 1, 4, 8, 9, 1, 4, 5, 9}, {1, 3, 6, 9, 1, 4, 7, 9, 1, 4, 6, 9},
    {1, 4, 7, 8}, {2, 3, 6, 8, 2, 5, 7, 8, 2, 5, 6, 8}, {2, 5, 7, 9}, {3, 6, 8, 9}, {}};

const std::vector<std::vector<size_t> > tet_table4D_tet_prism = {
    {}, {3, 6, 8, 9},
    {2, 5, 7, 9}, {2, 5, 7, 3, 6, 8}, {1, 4, 7, 8},
    {1, 4, 7, 3, 6, 9}, {1, 4, 8, 2, 5, 9}, {1, 2, 3, 4, 5, 6},
    {0, 4, 5, 6}, {0, 4, 5, 3, 8, 9}, {0, 4, 6, 2, 7, 9}, {0, 2, 3, 4, 7, 8},
    {0, 5, 6, 1, 7, 8}, {0, 1, 3, 5, 7, 9}, {0, 1, 2, 6, 8, 9}, {0, 1, 2, 3},
    {0, 1, 2, 3}, {0, 1, 2, 6, 8, 9}, {0, 1, 3, 5, 7, 9}, {0, 5, 6, 1, 7, 8},
    {0, 2, 3, 4, 7, 8}, {0, 4, 6, 2, 7, 9}, {0, 4, 5, 3, 8, 9}, {0, 4, 5, 6},
    {1, 2, 3, 4, 5, 6}, {1, 4, 8, 2, 5, 9}, {1, 4, 7, 3, 6, 9}, {1, 4, 7, 8},
    {2, 5, 7, 3, 6, 8}, {2, 5, 7, 9}, {3, 6, 8, 9}, {}
};

// lookup table for marching tets to compute upper level set
constexpr std::array<std::array<size_t, 2>, 10> tet_edges3D_extended = {
    {
        {{0, 0}}, {{1, 1}}, {{2, 2}}, {{3, 3}}, {{0, 1}},
        {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}
    }
};

const std::vector<std::vector<size_t> > tet_upper3D_verts = {
    {}, {3, 6, 8, 9}, {2, 5, 7, 9}, {2, 3, 5, 6, 7, 8}, {1, 4, 7, 8},
    {1, 3, 4, 6, 7, 9}, {1, 2, 4, 5, 8, 9}, {1, 2, 3, 4, 5, 6}, {0, 4, 5, 6},
    {0, 3, 4, 5, 8, 9}, {0, 2, 4, 6, 7, 9}, {0, 2, 3, 4, 7, 8},
    {0, 1, 5, 6, 7, 8}, {0, 1, 3, 5, 7, 9}, {0, 1, 2, 6, 8, 9}, {0, 1, 2, 3}};

// every 4 indices represent a tetrahedron
const std::vector<std::vector<size_t> > tet_upper3D_tets = {
    {}, {3, 6, 8, 9}, {2, 5, 7, 9}, {2, 5, 7, 8, 2, 5, 6, 8, 2, 3, 6, 8},
    {1, 4, 7, 8}, {1, 4, 7, 9, 1, 4, 6, 9, 1, 3, 6, 9},
    {1, 4, 8, 9, 1, 4, 5, 9, 1, 2, 5, 9}, {1, 4, 5, 6, 1, 2, 5, 6, 1, 2, 3, 6},
    {0, 4, 5, 6}, {0, 4, 5, 9, 0, 4, 8, 9, 0, 3, 8, 9},
    {0, 4, 6, 9, 0, 4, 7, 9, 0, 2, 7, 9}, {0, 8, 2, 3, 0, 8, 2, 7, 0, 8, 4, 7},
    {0, 5, 6, 8, 0, 5, 7, 8, 0, 1, 7, 8}, {0, 5, 7, 9, 0, 1, 7, 9, 0, 1, 3, 9},
    {0, 6, 8, 9, 0, 1, 8, 9, 0, 1, 2, 9}, {0, 1, 2, 3}};

const std::vector<std::vector<size_t> > tet_upper3D_prism = {
    {}, {3, 6, 9, 8}, {2, 5, 7, 9}, {3, 6, 8, 2, 5, 7}, {1, 7, 4, 8},
    {3, 9, 6, 1, 7, 4}, {1, 8, 4, 2, 9, 5}, {1, 2, 3, 4, 5, 6},
    {0, 4, 5, 6}, {3, 8, 9, 0, 4, 5}, {0, 4, 6, 2, 7, 9}, {4, 7, 8, 0, 2, 3},
    {0, 6, 5, 1, 8, 7}, {0, 1, 3, 5, 7, 9}, {0, 1, 2, 6, 8, 9}, {0, 1, 2, 3}
};

// lookup table for marching tets
constexpr std::array<std::array<size_t, 2>, 6> tet_edges3D = {
    {
        {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}
    }
};

const std::vector<std::vector<size_t> > tet_table3D_edges = {
    {}, {2, 4, 5}, {1, 3, 5}, {1, 2, 3, 4}, {0, 3, 4}, {0, 2, 3, 5},
    {0, 1, 4, 5}, {0, 1, 2}, {0, 1, 2}, {0, 1, 4, 5}, {0, 2, 3, 5},
    {0, 3, 4}, {1, 2, 3, 4}, {1, 3, 5}, {2, 4, 5}, {}
};

const std::vector<std::vector<size_t> > tet_table3d_tris = {
    {}, {2, 4, 5}, {1, 5, 3}, {1, 2, 4, 1, 4, 3}, {0, 3, 4},
    {0, 5, 2, 0, 3, 5}, {0, 1, 5, 0, 5, 4}, {0, 1, 2}, {0, 2, 1},
    {0, 5, 1, 0, 4, 5}, {0, 2, 5, 0, 5, 3}, {0, 4, 3},
    {1, 4, 2, 1, 3, 4}, {1, 3, 5}, {2, 5, 4}, {}
};

// lookup table for marching prisms
constexpr std::array<std::array<size_t, 2>, 9> prism_edges = {
    {
        {{0, 1}}, {{1, 2}}, {{0, 2}}, {{3, 4}},
        {{4, 5}}, {{3, 5}}, {{0, 3}}, {{1, 4}}, {{2, 5}}
    }
};

const std::vector<std::vector<size_t> > prism_table_edges = {
    {}, {4, 5, 8}, {3, 4, 7}, {3, 5, 7, 8}, {3, 5, 6}, {3, 4, 6, 8},
    {4, 5, 6, 7}, {6, 7, 8}, {1, 2, 8}, {1, 2, 4, 5}, {1, 2, 3, 4, 7, 8},
    {1, 2, 3, 5, 7}, {1, 2, 3, 5, 6, 8}, {1, 2, 3, 4, 6}, {1, 2, 4, 5, 6, 7, 8},
    {1, 2, 6, 7}, {0, 1, 7}, {0, 1, 4, 5, 7, 8}, {0, 1, 3, 4},
    {0, 1, 3, 5, 8}, {0, 1, 3, 5, 6, 7}, {0, 1, 3, 4, 6, 7, 8},
    {0, 1, 4, 5, 6}, {0, 1, 6, 8}, {0, 2, 7, 8}, {0, 2, 4, 5, 7},
    {0, 2, 3, 4, 8}, {0, 2, 3, 5}, {0, 2, 3, 5, 6, 7, 8}, {0, 2, 3, 4, 6, 7},
    {0, 2, 4, 5, 6, 8}, {0, 2, 6}, {0, 2, 6}, {0, 2, 4, 5, 6, 8},
    {0, 2, 3, 4, 6, 7}, {0, 2, 3, 5, 6, 7, 8}, {0, 2, 3, 5}, {0, 2, 3, 4, 8},
    {0, 2, 4, 5, 7}, {0, 2, 7, 8}, {0, 1, 6, 8}, {0, 1, 4, 5, 6},
    {0, 1, 3, 4, 6, 7, 8}, {0, 1, 3, 5, 6, 7}, {0, 1, 3, 5, 8}, {0, 1, 3, 4},
    {0, 1, 4, 5, 7, 8}, {0, 1, 7}, {1, 2, 6, 7}, {1, 2, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 6}, {1, 2, 3, 5, 6, 8}, {1, 2, 3, 5, 7}, {1, 2, 3, 4, 7, 8},
    {1, 2, 4, 5}, {1, 2, 8}, {6, 7, 8}, {4, 5, 6, 7}, {3, 4, 6, 8},
    {3, 5, 6}, {3, 5, 7, 8}, {3, 4, 7}, {4, 5, 8},
    {}
};

const std::vector<std::vector<size_t> > prism_table_tris = {
    {}, {4, 8, 5}, {3, 7, 4}, {3, 7, 8, 3, 8, 5}, {3, 5, 6}, {3, 4, 8, 3, 8, 6},
    {4, 5, 6, 4, 6, 7}, {6, 7, 8}, {1, 2, 8}, {1, 2, 5, 1, 5, 4}, {1, 2, 8, 3, 7, 4},
    {1, 2, 5, 1, 5, 7, 5, 3, 7}, {1, 2, 8, 3, 5, 6}, {1, 2, 6, 1, 6, 3, 1, 3, 4},
    {1, 2, 8, 4, 5, 6, 4, 6, 7}, {1, 2, 6, 1, 6, 7}, {0, 1, 7}, {0, 1, 7, 4, 8, 5},
    {0, 1, 4, 0, 4, 3}, {0, 1, 8, 0, 8, 5, 0, 5, 3}, {0, 1, 7, 3, 5, 6},
    {0, 1, 7, 3, 4, 8, 3, 8, 6}, {0, 1, 4, 0, 4, 6, 4, 5, 6}, {0, 1, 8, 0, 8, 6},
    {0, 2, 8, 0, 8, 7}, {0, 2, 5, 0, 5, 7, 5, 4, 7}, {0, 2, 8, 0, 8, 4, 0, 4, 3},
    {0, 2, 5, 0, 5, 3}, {0, 2, 8, 0, 8, 7, 3, 5, 6}, {0, 2, 7, 2, 4, 7, 2, 6, 4, 6, 3, 4},
    {0, 2, 8, 0, 8, 4, 0, 4, 6, 4, 5, 6}, {0, 2, 6}, {0, 6, 2}, {0, 6, 2, 4, 8, 5},
    {0, 6, 2, 3, 7, 4}, {0, 6, 2, 3, 7, 8, 3, 8, 5}, {0, 3, 5, 0, 5, 2}, {0, 3, 4, 0, 4, 8, 0, 8, 2},
    {0, 7, 5, 7, 4, 5, 0, 5, 2}, {0, 7, 8, 0, 8, 2}, {0, 6, 8, 0, 8, 1}, {0, 6, 4, 6, 5, 4, 0, 4, 1},
    {0, 6, 8, 0, 8, 1, 3, 7, 4}, {0, 6, 1, 6, 5, 1, 5, 7, 1, 5, 3, 7}, {0, 3, 5, 0, 5, 8, 0, 8, 1},
    {0, 3, 4, 0, 4, 1}, {0, 7, 5, 7, 4, 5, 0, 5, 8, 0, 8, 1}, {0, 7, 1}, {1, 7, 6, 1, 6, 2},
    {1, 7, 6, 1, 6, 2, 4, 8, 5}, {1, 4, 3, 1, 3, 6, 1, 6, 2}, {1, 8, 3, 8, 5, 3, 1, 3, 6, 1, 6, 2},
    {1, 7, 5, 7, 3, 5, 1, 5, 2}, {1, 7, 2, 7, 3, 2, 3, 8, 2, 3, 4, 8}, {1, 4, 5, 1, 5, 2},
    {1, 8, 2}, {6, 8, 7}, {4, 7, 6, 4, 6, 5}, {3, 6, 8, 3, 8, 4}, {3, 6, 5},
    {3, 5, 8, 3, 8, 7}, {3, 4, 7}, {4, 5, 8}, {}
};

const std::vector<std::vector<std::vector<size_t> > > prism_table_cycles = {
    {}, {{4, 8, 5}}, {{3, 7, 4}}, {{3, 7, 8, 5}}, {{3, 5, 6}},
    {{3, 4, 8, 6}}, {{4, 5, 6, 7}}, {{6, 7, 8}}, {{1, 2, 8}},
    {{1, 2, 5, 4}}, {{1, 2, 8}, {3, 7, 4}}, {{1, 2, 5, 3, 7}},
    {{1, 2, 8}, {3, 5, 6}}, {{1, 2, 6, 3, 4}}, {{1, 2, 8}, {4, 5, 6, 7}},
    {{1, 2, 6, 7}}, {{0, 1, 7}}, {{0, 1, 7}, {4, 8, 5}}, {{0, 1, 4, 3}},
    {{0, 1, 8, 5, 3}}, {{0, 1, 7}, {3, 5, 6}}, {{0, 1, 7}, {3, 4, 8, 6}},
    {{0, 1, 4, 5, 6}}, {{0, 1, 8, 6}}, {{0, 2, 8, 7}}, {{0, 2, 5, 4, 7}},
    {{0, 2, 8, 4, 3}}, {{0, 2, 5, 3}}, {{0, 2, 8, 7}, {3, 5, 6}},
    {{0, 2, 6, 3, 4, 7}}, {{0, 2, 8, 4, 5, 6}}, {{0, 2, 6}}, {{0, 6, 2}},
    {{0, 6, 2}, {4, 8, 5}}, {{0, 6, 2}, {3, 7, 4}}, {{0, 6, 2}, {3, 7, 8, 5}},
    {{0, 3, 5, 2}}, {{0, 3, 4, 8, 2}}, {{0, 7, 4, 5, 2}}, {{0, 7, 8, 2}},
    {{0, 6, 8, 1}}, {{0, 6, 5, 4, 1}}, {{0, 6, 8, 1}, {3, 7, 4}},
    {{0, 6, 5, 3, 7, 1}}, {{0, 3, 5, 8, 1}}, {{0, 3, 4, 1}}, {{0, 7, 4, 5, 8, 1}},
    {{0, 7, 1}}, {{1, 7, 6, 2}}, {{1, 7, 6, 2}, {4, 8, 5}}, {{1, 4, 3, 6, 2}},
    {{1, 8, 5, 3, 6, 2}}, {{1, 7, 3, 5, 2}}, {{1, 7, 3, 4, 8, 2}}, {{1, 4, 5, 2}},
    {{1, 8, 2}}, {{6, 8, 7}}, {{4, 7, 6, 5}}, {{3, 6, 8, 4}}, {{3, 6, 5}},
    {{3, 5, 8, 7}}, {{3, 4, 7}}, {{4, 5, 8}}, {}
};

// Custom hash function for a std::pair
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);

    // Combine the two hash values
    return h1 ^ (h2 << 1); // XOR with bit-shift to mix bits
  }
};

template <typename Float, size_t N>
void get_linear_root(const std::array<Float, N>& p1, const std::array<Float, N>& p2,
                     const Float val1, const Float val2, std::array<Float, N>& output) {
  Float t = val1 / (val1 - val2);
  for (size_t i = 0; i < N; i++) {
    output[i] = p1[i] + t * (p2[i] - p1[i]);
  }
}

// compute a cubic root in (0,1) given values and gradients at two endpoints
template <typename Float>
Float get_cubic_root(Float val1, Float val2, Float g1, Float g2) {
  assert(val1 * val2 < 0);

  // make sure val1 < 0 and val2 > 0
  if (val1 > 0) {
    val1 = -val1;
    val2 = -val2;
    g1 = -g1;
    g2 = -g2;
  }

  // compute the cubic function f(x) = a*x^3 + b*x^2 + c*x + d
  const Float a = g1 + g2 + 2 * (val1 - val2);
  const Float b = 3 * (val2 - val1) - 2 * g1 - g2;
  const Float c = g1;
  const Float d = val1;

  // initial guess: the linear root
  Float x = val1 / (val1 - val2);

  // root finding: combine Halley's method and bisect method
  // mostly a bisect method, but first find the next guess using Hally's method,
  // if the guess doesn't lie in the sign-changing interval, use the midpoint of that interval
  // terminate when the change in x is small
  Float xlo = 0;
  Float xhi = 1;
  constexpr Float x_tol = 1e-4;
  while (true) {
    Float f = d + x * (c + x * (b + x * a));
    if (f == 0) {
      break;
    }
    if (f < 0) {
      xlo = x;
    } else {
      xhi = x;
    }
    // f'(x) = 3*a*x^2 + 2*b*x + c
    // f''(x) = 6*a*x + 2*b
    Float df = c + x * (2 * b + x * 3 * a);
    Float ddf = 2 * b + x * 6 * a;
    Float dx = 2 * f * df / (2 * df * df - f * ddf);
    Float x_new = x - dx;
    if (x_new <= xlo || x_new >= xhi) {
      x_new = 0.5 * (xlo + xhi);
    }
    x = x_new;
    if (std::abs(dx) < x_tol) {
      break;
    }
  }

  return x;
}

template <typename Float>
void get_cubic_root(const std::array<Float, 4>& p1, const std::array<Float, 4>& p2,
                    const Float val1, const Float val2, const std::array<Float, 4>& grad1,
                    const std::array<Float, 4>& grad2,
                    std::array<Float, 3>& output) {
  // require val1 and val2 to have different signs
  assert(val1 * val2 < 0);

  // directional derivative
  const std::array<Float, 4> p1p2 = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2], p2[3] - p1[3]};
  // since the interval is scaled to (0,1), we don't normalize by dividing by the length of p1p2
  const Float g1 = (grad1[0] * p1p2[0] + grad1[1] * p1p2[1] + grad1[2] * p1p2[2] + grad1[3] * p1p2[3]);
  const Float g2 = (grad2[0] * p1p2[0] + grad2[1] * p1p2[1] + grad2[2] * p1p2[2] + grad2[3] * p1p2[3]);

  // compute the root
  const Float x = get_cubic_root(val1, val2, g1, g2);
  output[0] = p1[0] + x * p1p2[0];
  output[1] = p1[1] + x * p1p2[1];
  output[2] = p1[2] + x * p1p2[2];
}

template <typename Float>
void get_cubic_root(const std::array<Float, 3>& p1, const std::array<Float, 3>& p2,
                    const Float val1, const Float val2, const std::array<Float, 3>& grad1,
                    const std::array<Float, 3>& grad2,
                    std::array<Float, 3>& output) {
    // require val1 and val2 to have different signs
    assert(val1 * val2 < 0);
    
    // directional derivative
    const std::array<Float, 3> p1p2 = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    // since the interval is scaled to (0,1), we don't normalize by dividing by the length of p1p2
    const Float g1 = (grad1[0] * p1p2[0] + grad1[1] * p1p2[1] + grad1[2] * p1p2[2]);
    const Float g2 = (grad2[0] * p1p2[0] + grad2[1] * p1p2[1] + grad2[2] * p1p2[2]);
    
    // compute the root
    const Float x = get_cubic_root(val1, val2, g1, g2);
    output[0] = p1[0] + x * p1p2[0];
    output[1] = p1[1] + x * p1p2[1];
    output[2] = p1[2] + x * p1p2[2];
}

template <typename Float>
int get_tet_orient(const std::array<Float, 3>& a, const std::array<Float, 3>& b, const std::array<Float, 3>& c,
                   const std::array<Float, 3>& d) {
  const Float det = (b[0] - a[0]) * ((c[1] - a[1]) * (d[2] - a[2]) - (c[2] - a[2]) * (d[1] - a[1])) -
                    (b[1] - a[1]) * ((c[0] - a[0]) * (d[2] - a[2]) - (c[2] - a[2]) * (d[0] - a[0])) +
                    (b[2] - a[2]) * ((c[0] - a[0]) * (d[1] - a[1]) - (c[1] - a[1]) * (d[0] - a[0]));
  return det > 0 ? 1 : (det < 0 ? -1 : 0);
}

template <typename Float>
int get_prism_orient(const std::array<Float, 3>& p1, const std::array<Float, 3>& p2,
                     const std::array<Float, 3>& p3,
                     const std::array<Float, 3>& p4, const std::array<Float, 3>& p5,
                     const std::array<Float, 3>& p6) {
  const std::array<Float, 3> top = {
      (p4[0] + p5[0] + p6[0]) / 3, (p4[1] + p5[1] + p6[1]) / 3, (p4[2] + p5[2] + p6[2]) / 3
  };
  const int orient = get_tet_orient(p1, p2, p3, top);
  if (orient == 0) {
    const std::array<Float, 3> bottom = {
        (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3, (p1[2] + p2[2] + p3[2]) / 3
    };
    return -get_tet_orient(p4, p5, p6, bottom);
  }
  return orient;
}
}

namespace marching3D {
template <typename Float>
void Marching4Simplex(const std::vector<std::array<Float, 4> >& vertices,
                      const std::vector<std::array<size_t, 5> >& simplices, const std::vector<Float>& values,
                      std::vector<std::array<Float, 4> >& output_vertices,
                      std::vector<std::array<size_t, 4> >& output_tets) {
  assert(vertices.size() == values.size());
  output_vertices.clear();
  output_tets.clear();

  // map edge to output vertex index
  std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> edge_map;

  // find if the edge is already processed, if not, create a new vertex
  auto get_new_vertex = [&edge_map, &output_vertices, &vertices, &values](size_t v1, size_t v2) {
    if (v1 > v2) {
      std::swap(v1, v2);
    }
    if (const auto edge = std::make_pair(v1, v2); edge_map.find(edge) == edge_map.end()) {
      edge_map[edge] = output_vertices.size();
      auto& new_vertex = output_vertices.emplace_back();
      get_linear_root(vertices[v1], vertices[v2], values[v1], values[v2], new_vertex);
      return output_vertices.size() - 1;
    } else {
      return edge_map[edge];
    }
  };

  // process each simplex
  std::vector<size_t> local_edge_map(10);
  for (const auto& simplex : simplices) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 5; j++) {
      if (values[simplex[4 - j]] > 0) {
        index |= 1 << j;
      }
    }

    // add new vertices
    const auto& tet_verts = tet_table4D_verts[index];
    if (tet_verts.empty()) {
      continue;
    }
    for (auto j : tet_verts) {
      auto [v1, v2] = tet_edges4D[j];
      v1 = simplex[v1];
      v2 = simplex[v2];
      local_edge_map[j] = get_new_vertex(v1, v2);
    }

    // add new tets
    const auto& tet_tets = tet_table4D_tets[index];
    for (size_t j = 0; j < tet_tets.size(); j += 4) {
      auto& new_tet = output_tets.emplace_back();
      for (size_t k = 0; k < 4; k++) {
        new_tet[k] = local_edge_map[tet_tets[j + k]];
      }
    }
  }
}

template void Marching4Simplex<double>(const std::vector<std::array<double, 4> >& vertices,
                                       const std::vector<std::array<size_t, 5> >& simplices,
                                       const std::vector<double>& values,
                                       std::vector<std::array<double, 4> >& output_vertices,
                                       std::vector<std::array<size_t, 4> >& output_tets);

template void Marching4Simplex<float>(const std::vector<std::array<float, 4> >& vertices,
                                      const std::vector<std::array<size_t, 5> >& simplices,
                                      const std::vector<float>& values,
                                      std::vector<std::array<float, 4> >& output_vertices,
                                      std::vector<std::array<size_t, 4> >& output_tets);

template <typename Float>
void Marching4Simplex(const std::vector<std::array<Float, 4> >& vertices,
                      const std::vector<std::array<size_t, 5> >& simplices, const std::vector<Float>& values,
                      std::vector<std::array<Float, 4> >& output_vertices,
                      std::vector<std::array<size_t, 4> >& output_tets,
                      std::vector<std::array<size_t, 6> >& output_prisms) {
  assert(vertices.size() == values.size());
  output_vertices.clear();
  output_tets.clear();
  output_prisms.clear();

  // map edge to output vertex index
  std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> edge_map;

  // find if the edge is already processed, if not, create a new vertex
  auto get_new_vertex = [&edge_map, &output_vertices, &vertices, &values](size_t v1, size_t v2) {
    if (v1 > v2) {
      std::swap(v1, v2);
    }
    if (const auto edge = std::make_pair(v1, v2); edge_map.find(edge) == edge_map.end()) {
      edge_map[edge] = output_vertices.size();
      auto& new_vertex = output_vertices.emplace_back();
      get_linear_root(vertices[v1], vertices[v2], values[v1], values[v2], new_vertex);
      return output_vertices.size() - 1;
    } else {
      return edge_map[edge];
    }
  };

  // process each simplex
  for (const auto& simplex : simplices) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 5; j++) {
      if (values[simplex[4 - j]] > 0) {
        index |= 1 << j;
      }
    }

    // determine the type of the simplex
    const std::vector<size_t>& tet_prism = tet_table4D_tet_prism[index];
    if (tet_prism.empty()) {
      continue;
    }
    if (tet_prism.size() == 4) {
      // create a new tet
      auto& tet = output_tets.emplace_back();
      for (size_t j = 0; j < 4; j++) {
        auto [v1, v2] = tet_edges4D[tet_prism[j]];
        v1 = simplex[v1];
        v2 = simplex[v2];
        tet[j] = get_new_vertex(v1, v2);
      }
    } else {
      // create a new prism
      auto& prism = output_prisms.emplace_back();
      for (size_t j = 0; j < 6; j++) {
        auto [v1, v2] = tet_edges4D[tet_prism[j]];
        v1 = simplex[v1];
        v2 = simplex[v2];
        prism[j] = get_new_vertex(v1, v2);
      }
    }
  }
}

template void Marching4Simplex<double>(const std::vector<std::array<double, 4> >& vertices,
                                       const std::vector<std::array<size_t, 5> >& simplices,
                                       const std::vector<double>& values,
                                       std::vector<std::array<double, 4> >& output_vertices,
                                       std::vector<std::array<size_t, 4> >& output_tets,
                                       std::vector<std::array<size_t, 6> >& output_prisms);

template void Marching4Simplex<float>(const std::vector<std::array<float, 4> >& vertices,
                                      const std::vector<std::array<size_t, 5> >& simplices,
                                      const std::vector<float>& values,
                                      std::vector<std::array<float, 4> >& output_vertices,
                                      std::vector<std::array<size_t, 4> >& output_tets,
                                      std::vector<std::array<size_t, 6> >& output_prisms);

template <typename Float>
void Marching4SimplexUL(const std::vector<std::array<Float, 4> >& vertices,
                        const std::vector<std::array<size_t, 5> >& simplices,
                        const std::vector<std::array<size_t, 4> >& upper_tets,
                        const std::vector<std::array<size_t, 4> >& lower_tets, const std::vector<Float>& values,
                        std::vector<std::array<Float, 4> >& output_vertices,
                        std::vector<std::array<size_t, 4> >& output_tets) {
  assert(vertices.size() == values.size());
  output_vertices.clear();
  output_tets.clear();

  // map edge to output vertex index
  std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> edge_map;

  // find if the edge is already processed, if not, create a new vertex
  auto get_new_vertex = [&edge_map, &output_vertices, &vertices, &values](size_t v1, size_t v2) {
    if (v1 > v2) {
      std::swap(v1, v2);
    }
    if (const auto edge = std::make_pair(v1, v2); edge_map.find(edge) == edge_map.end()) {
      edge_map[edge] = output_vertices.size();
      auto& new_vertex = output_vertices.emplace_back();
      if (v1 == v2) {
        new_vertex = vertices[v1];
      } else {
        get_linear_root(vertices[v1], vertices[v2], values[v1], values[v2], new_vertex);
      }
      return output_vertices.size() - 1;
    } else {
      return edge_map[edge];
    }
  };

  // process each simplex
  std::vector<size_t> local_edge_map(10);
  for (const auto& simplex : simplices) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 5; j++) {
      if (values[simplex[4 - j]] > 0) {
        index |= 1 << j;
      }
    }

    // add new vertices
    const auto& tet_verts = tet_table4D_verts[index];
    if (tet_verts.empty()) {
      continue;
    }
    for (auto j : tet_verts) {
      auto [v1, v2] = tet_edges4D[j];
      v1 = simplex[v1];
      v2 = simplex[v2];
      local_edge_map[j] = get_new_vertex(v1, v2);
    }

    // add new tets
    const auto& tet_tets = tet_table4D_tets[index];
    for (size_t j = 0; j < tet_tets.size(); j += 4) {
      auto& new_tet = output_tets.emplace_back();
      for (size_t k = 0; k < 4; k++) {
        new_tet[k] = local_edge_map[tet_tets[j + k]];
      }
    }
  }

  // process upper tets
  for (const auto& tet : upper_tets) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 4; j++) {
      if (values[tet[3 - j]] > 0) {
        index |= 1 << j;
      }
    }

    // add new vertices
    const auto& tet_verts = tet_upper3D_verts[index];
    if (tet_verts.empty()) {
      continue;
    }
    for (unsigned long j : tet_verts) {
      auto [v1, v2] = tet_edges3D_extended[j];
      v1 = tet[v1];
      v2 = tet[v2];
      local_edge_map[j] = get_new_vertex(v1, v2);
    }

    // add new tets
    const auto& tet_tets = tet_upper3D_tets[index];
    for (size_t j = 0; j < tet_tets.size(); j += 4) {
      auto& new_tet = output_tets.emplace_back();
      for (size_t k = 0; k < 4; k++) {
        new_tet[k] = local_edge_map[tet_tets[j + k]];
      }
    }
  }

  // process lower tets
  for (const auto& tet : lower_tets) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 4; j++) {
      if (values[tet[3 - j]] < 0) {
        index |= 1 << j;
      }
    }

    // add new vertices
    const auto& tet_verts = tet_upper3D_verts[index];
    if (tet_verts.empty()) {
      continue;
    }
    for (auto j : tet_verts) {
      auto [v1, v2] = tet_edges3D_extended[j];
      v1 = tet[v1];
      v2 = tet[v2];
      local_edge_map[j] = get_new_vertex(v1, v2);
    }

    // add new tets
    const auto& tet_tets = tet_upper3D_tets[index];
    for (size_t j = 0; j < tet_tets.size(); j += 4) {
      auto& new_tet = output_tets.emplace_back();
      for (size_t k = 0; k < 4; k++) {
        new_tet[k] = local_edge_map[tet_tets[j + k]];
      }
    }
  }
}

template void Marching4SimplexUL<double>(const std::vector<std::array<double, 4> >& vertices,
                                         const std::vector<std::array<size_t, 5> >& simplices,
                                         const std::vector<std::array<size_t, 4> >& upper_tets,
                                         const std::vector<std::array<size_t, 4> >& lower_tets,
                                         const std::vector<double>& values,
                                         std::vector<std::array<double, 4> >& output_vertices,
                                         std::vector<std::array<size_t, 4> >& output_tets);

template void Marching4SimplexUL<float>(const std::vector<std::array<float, 4> >& vertices,
                                        const std::vector<std::array<size_t, 5> >& simplices,
                                        const std::vector<std::array<size_t, 4> >& upper_tets,
                                        const std::vector<std::array<size_t, 4> >& lower_tets,
                                        const std::vector<float>& values,
                                        std::vector<std::array<float, 4> >& output_vertices,
                                        std::vector<std::array<size_t, 4> >& output_tets);


template <typename Float>
void Marching4SimplexUL(const std::vector<std::array<Float, 4> >& vertices,
                        const std::vector<std::array<size_t, 5> >& simplices,
                        const std::vector<std::array<size_t, 4> >& upper_tets,
                        const std::vector<std::array<size_t, 4> >& lower_tets, const std::vector<Float>& values,
                        std::vector<std::array<Float, 4> >& output_vertices,
                        std::vector<std::array<size_t, 4> >& output_tets,
                        std::vector<std::array<size_t, 6> >& output_prisms) {
  assert(vertices.size() == values.size());
  output_vertices.clear();
  output_tets.clear();
  output_prisms.clear();

  // map edge to output vertex index
  std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> edge_map;

  // find if the edge is already processed, if not, create a new vertex
  auto get_new_vertex = [&edge_map, &output_vertices, &vertices, &values](size_t v1, size_t v2) {
    if (v1 > v2) {
      std::swap(v1, v2);
    }
    if (const auto edge = std::make_pair(v1, v2); edge_map.find(edge) == edge_map.end()) {
      edge_map[edge] = output_vertices.size();
      auto& new_vertex = output_vertices.emplace_back();
      if (v1 == v2) {
        new_vertex = vertices[v1];
      } else {
        get_linear_root(vertices[v1], vertices[v2], values[v1], values[v2], new_vertex);
      }
      return output_vertices.size() - 1;
    } else {
      return edge_map[edge];
    }
  };

  // process each simplex
  for (const auto& simplex : simplices) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 5; j++) {
      if (values[simplex[4 - j]] > 0) {
        index |= 1 << j;
      }
    }

    // determine the type of the simplex
    const std::vector<size_t>& tet_prism = tet_table4D_tet_prism[index];
    if (tet_prism.empty()) {
      continue;
    }
    if (tet_prism.size() == 4) {
      // create a new tet
      auto& tet = output_tets.emplace_back();
      for (size_t j = 0; j < 4; j++) {
        auto [v1, v2] = tet_edges4D[tet_prism[j]];
        v1 = simplex[v1];
        v2 = simplex[v2];
        tet[j] = get_new_vertex(v1, v2);
      }
    } else {
      // create a new prism
      auto& prism = output_prisms.emplace_back();
      for (size_t j = 0; j < 6; j++) {
        auto [v1, v2] = tet_edges4D[tet_prism[j]];
        v1 = simplex[v1];
        v2 = simplex[v2];
        prism[j] = get_new_vertex(v1, v2);
      }
    }
  }

  // process upper tets
  for (const auto& tet : upper_tets) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 4; j++) {
      if (values[tet[3 - j]] > 0) {
        index |= 1 << j;
      }
    }

    const std::vector<size_t>& tet_prism = tet_upper3D_prism[index];
    if (tet_prism.empty()) {
      continue;
    }
    if (tet_prism.size() == 4) {
      // create a new tet
      auto& new_tet = output_tets.emplace_back();
      for (size_t j = 0; j < 4; j++) {
        auto [v1, v2] = tet_edges3D_extended[tet_prism[j]];
        v1 = tet[v1];
        v2 = tet[v2];
        new_tet[j] = get_new_vertex(v1, v2);
      }
    } else {
      // create a new prism
      auto& new_prism = output_prisms.emplace_back();
      for (size_t j = 0; j < 6; j++) {
        auto [v1, v2] = tet_edges3D_extended[tet_prism[j]];
        v1 = tet[v1];
        v2 = tet[v2];
        new_prism[j] = get_new_vertex(v1, v2);
      }
    }
  }

  // process lower tets
  for (const auto& tet : lower_tets) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 4; j++) {
      if (values[tet[3 - j]] < 0) {
        index |= 1 << j;
      }
    }

    const std::vector<size_t>& tet_prism = tet_upper3D_prism[index];
    if (tet_prism.empty()) {
      continue;
    }
    if (tet_prism.size() == 4) {
      // create a new tet
      auto& new_tet = output_tets.emplace_back();
      for (size_t j = 0; j < 4; j++) {
        auto [v1, v2] = tet_edges3D_extended[tet_prism[j]];
        v1 = tet[v1];
        v2 = tet[v2];
        new_tet[j] = get_new_vertex(v1, v2);
      }
    } else {
      // create a new prism
      auto& new_prism = output_prisms.emplace_back();
      for (size_t j = 0; j < 6; j++) {
        auto [v1, v2] = tet_edges3D_extended[tet_prism[j]];
        v1 = tet[v1];
        v2 = tet[v2];
        new_prism[j] = get_new_vertex(v1, v2);
      }
    }
  }
}

template void Marching4SimplexUL<double>(const std::vector<std::array<double, 4> >& vertices,
                                         const std::vector<std::array<size_t, 5> >& simplices,
                                         const std::vector<std::array<size_t, 4> >& upper_tets,
                                         const std::vector<std::array<size_t, 4> >& lower_tets,
                                         const std::vector<double>& values,
                                         std::vector<std::array<double, 4> >& output_vertices,
                                         std::vector<std::array<size_t, 4> >& output_tets,
                                         std::vector<std::array<size_t, 6> >& output_prisms);

template void Marching4SimplexUL<float>(const std::vector<std::array<float, 4> >& vertices,
                                        const std::vector<std::array<size_t, 5> >& simplices,
                                        const std::vector<std::array<size_t, 4> >& upper_tets,
                                        const std::vector<std::array<size_t, 4> >& lower_tets,
                                        const std::vector<float>& values,
                                        std::vector<std::array<float, 4> >& output_vertices,
                                        std::vector<std::array<size_t, 4> >& output_tets,
                                        std::vector<std::array<size_t, 6> >& output_prisms);

template <typename Float>
void Marching4SimplexCyclic(const std::vector<std::array<Float, 4> >& vertices,
                            const std::vector<std::array<size_t, 5> >& simplices,
                            const std::vector<std::array<size_t, 2> >& vertex_pairs,
                            const std::vector<Float>& values, std::vector<std::array<Float, 4> >& output_vertices,
                            std::vector<std::array<size_t, 4> >& output_tets) {
  assert(vertices.size() == values.size());
  output_vertices.clear();
  output_tets.clear();

  std::vector<int> signs(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    signs[i] = values[i] > 0 ? 1 : 0;
  }

  constexpr size_t unpaired = std::numeric_limits<size_t>::max();
  std::vector<size_t> paired_vertices(vertices.size(), unpaired);
  for (const auto& pair : vertex_pairs) {
    paired_vertices[pair[0]] = pair[1];
    signs[pair[0]] = signs[pair[1]];
  }

  // map edge to output vertex index
  std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> edge_map;

  // find if the edge is already processed, if not, create a new vertex
  auto get_new_vertex = [&edge_map, &output_vertices, &vertices, &values](size_t v1, size_t v2) {
    if (v1 > v2) {
      std::swap(v1, v2);
    }
    if (const auto edge = std::make_pair(v1, v2); edge_map.find(edge) == edge_map.end()) {
      edge_map[edge] = output_vertices.size();
      auto& new_vertex = output_vertices.emplace_back();
      get_linear_root(vertices[v1], vertices[v2], values[v1], values[v2], new_vertex);
      return output_vertices.size() - 1;
    } else {
      return edge_map[edge];
    }
  };

  // process each simplex
  std::vector<size_t> local_edge_map(10);
  for (const auto& simplex : simplices) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 5; j++) {
      if (signs[simplex[4 - j]] == 1) {
        index |= 1 << j;
      }
    }

    // add new vertices
    const auto& tet_verts = tet_table4D_verts[index];
    for (auto j : tet_verts) {
      auto [v1, v2] = tet_edges4D[j];
      v1 = simplex[v1];
      v2 = simplex[v2];
      if (paired_vertices[v1] != unpaired && paired_vertices[v2] != unpaired) {
        v1 = paired_vertices[v1];
        v2 = paired_vertices[v2];
      }
      local_edge_map[j] = get_new_vertex(v1, v2);
    }

    // add new tets
    const auto& tet_tets = tet_table4D_tets[index];
    for (size_t j = 0; j < tet_tets.size(); j += 4) {
      auto& new_tet = output_tets.emplace_back();
      for (size_t k = 0; k < 4; k++) {
        new_tet[k] = local_edge_map[tet_tets[j + k]];
      }
    }
  }
}

template void Marching4SimplexCyclic<double>(const std::vector<std::array<double, 4> >& vertices,
                                             const std::vector<std::array<size_t, 5> >& simplices,
                                             const std::vector<std::array<size_t, 2> >& vertex_pairs,
                                             const std::vector<double>& values,
                                             std::vector<std::array<double, 4> >& output_vertices,
                                             std::vector<std::array<size_t, 4> >& output_tets);

template void Marching4SimplexCyclic<float>(const std::vector<std::array<float, 4> >& vertices,
                                            const std::vector<std::array<size_t, 5> >& simplices,
                                            const std::vector<std::array<size_t, 2> >& vertex_pairs,
                                            const std::vector<float>& values,
                                            std::vector<std::array<float, 4> >& output_vertices,
                                            std::vector<std::array<size_t, 4> >& output_tets);

template <typename Float>
void Marching4SimplexCyclic(const std::vector<std::array<Float, 4> >& vertices,
                            const std::vector<std::array<size_t, 5> >& simplices,
                            const std::vector<std::array<size_t, 2> >& vertex_pairs,
                            const std::vector<Float>& values, std::vector<std::array<Float, 4> >& output_vertices,
                            std::vector<std::array<size_t, 4> >& output_tets,
                            std::vector<std::array<size_t, 6> >& output_prisms) {
  assert(vertices.size() == values.size());
  output_vertices.clear();
  output_tets.clear();
  output_prisms.clear();

  std::vector<int> signs(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    signs[i] = values[i] > 0 ? 1 : 0;
  }

  constexpr size_t unpaired = std::numeric_limits<size_t>::max();
  std::vector<size_t> paired_vertices(vertices.size(), unpaired);
  for (const auto& pair : vertex_pairs) {
    paired_vertices[pair[0]] = pair[1];
    signs[pair[0]] = signs[pair[1]];
  }

  // map edge to output vertex index
  std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> edge_map;

  // find if the edge is already processed, if not, create a new vertex
  auto get_new_vertex = [&edge_map, &output_vertices, &vertices, &values](size_t v1, size_t v2) {
    if (v1 > v2) {
      std::swap(v1, v2);
    }
    if (const auto edge = std::make_pair(v1, v2); edge_map.find(edge) == edge_map.end()) {
      edge_map[edge] = output_vertices.size();
      auto& new_vertex = output_vertices.emplace_back();
      get_linear_root(vertices[v1], vertices[v2], values[v1], values[v2], new_vertex);
      return output_vertices.size() - 1;
    } else {
      return edge_map[edge];
    }
  };

  // process each simplex
  for (const auto& simplex : simplices) {
    // determine the index into the table
    size_t index = 0;
    for (size_t j = 0; j < 5; j++) {
      if (signs[simplex[4 - j]] == 1) {
        index |= 1 << j;
      }
    }

    // determine the type of the simplex
    const std::vector<size_t>& tet_prism = tet_table4D_tet_prism[index];
    if (tet_prism.empty()) {
      continue;
    }
    if (tet_prism.size() == 4) {
      // create a new tet
      auto& tet = output_tets.emplace_back();
      for (size_t j = 0; j < 4; j++) {
        auto [v1, v2] = tet_edges4D[tet_prism[j]];
        v1 = simplex[v1];
        v2 = simplex[v2];
        if (paired_vertices[v1] != unpaired && paired_vertices[v2] != unpaired) {
          v1 = paired_vertices[v1];
          v2 = paired_vertices[v2];
        }
        tet[j] = get_new_vertex(v1, v2);
      }
    } else {
      // create a new prism
      auto& prism = output_prisms.emplace_back();
      for (size_t j = 0; j < 6; j++) {
        auto [v1, v2] = tet_edges4D[tet_prism[j]];
        v1 = simplex[v1];
        v2 = simplex[v2];
        if (paired_vertices[v1] != unpaired && paired_vertices[v2] != unpaired) {
          v1 = paired_vertices[v1];
          v2 = paired_vertices[v2];
        }
        prism[j] = get_new_vertex(v1, v2);
      }
    }
  }
}

template void Marching4SimplexCyclic<double>(const std::vector<std::array<double, 4> >& vertices,
                                             const std::vector<std::array<size_t, 5> >& simplices,
                                             const std::vector<std::array<size_t, 2> >& vertex_pairs,
                                             const std::vector<double>& values,
                                             std::vector<std::array<double, 4> >& output_vertices,
                                             std::vector<std::array<size_t, 4> >& output_tets,
                                             std::vector<std::array<size_t, 6> >& output_prisms);

template void Marching4SimplexCyclic<float>(const std::vector<std::array<float, 4> >& vertices,
                                            const std::vector<std::array<size_t, 5> >& simplices,
                                            const std::vector<std::array<size_t, 2> >& vertex_pairs,
                                            const std::vector<float>& values,
                                            std::vector<std::array<float, 4> >& output_vertices,
                                            std::vector<std::array<size_t, 4> >& output_tets,
                                            std::vector<std::array<size_t, 6> >& output_prisms);

template <typename Float>
void MarchingTetPrism(const std::vector<std::array<Float, 4> >& vertices,
                      const std::vector<std::array<size_t, 4> >& tets,
                      const std::vector<std::array<size_t, 6> >& prisms,
                      const std::vector<Float>& values, const std::vector<std::array<Float, 4> >& gradients,
                      std::vector<std::array<Float, 3> >& output_vertices,
                      std::vector<std::array<size_t, 3> >& output_triangles,
                      bool prism_insert_cycle_center) {
  assert(vertices.size() == values.size());
  const bool has_gradients = gradients.size() == vertices.size();
  output_vertices.clear();
  output_triangles.clear();

  // project vertices to 3D
  std::vector<std::array<Float, 3> > vertices3D(vertices.size());
  for (size_t i = 0; i < vertices.size(); i++) {
    for (size_t j = 0; j < 3; j++) {
      vertices3D[i][j] = vertices[i][j];
    }
  }

  // map edge to output vertex index
  std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> edge_map;

  // find if the edge is already processed, if not, create a new vertex
  auto get_new_vertex = [&edge_map, &output_vertices, &vertices, &vertices3D, &values, &gradients, &has_gradients
      ](size_t v1, size_t v2) {
    if (v1 > v2) {
      std::swap(v1, v2);
    }
    auto edge = std::make_pair(v1, v2);
    if (edge_map.find(edge) == edge_map.end()) {
      edge_map[edge] = output_vertices.size();
      auto& new_vertex = output_vertices.emplace_back();
      if (has_gradients) {
        get_cubic_root(vertices[v1], vertices[v2], values[v1], values[v2], gradients[v1], gradients[v2],
                       new_vertex);
      } else {
        get_linear_root(vertices3D[v1], vertices3D[v2], values[v1], values[v2], new_vertex);
      }
      return output_vertices.size() - 1;
    } else {
      return edge_map[edge];
    }
  };

  // process tets
  std::array<size_t, 6> tet_edge_vertices{};
  for (const auto& tet : tets) {
    // index
    size_t index = 0;
    for (size_t j = 0; j < 4; ++j) {
      if (values[tet[3 - j]] > 0) {
        index |= 1 << j;
      }
    }

    const auto& edges = tet_table3D_edges[index];
    if (edges.empty()) {
      continue;
    }
    for (const auto& e : edges) {
      auto [v1, v2] = tet_edges3D[e];
      v1 = tet[v1];
      v2 = tet[v2];
      tet_edge_vertices[e] = get_new_vertex(v1, v2);
    }

    // create new triangles
    const auto& tris = tet_table3d_tris[index];
    if (get_tet_orient(vertices3D[tet[0]], vertices3D[tet[1]], vertices3D[tet[2]], vertices3D[tet[3]]) == 1) {
      for (size_t j = 0; j < tris.size(); j += 3) {
        output_triangles.push_back({
            tet_edge_vertices[tris[j]], tet_edge_vertices[tris[j + 1]], tet_edge_vertices[tris[j + 2]]
        });
      }
    } else {
      for (size_t j = 0; j < tris.size(); j += 3) {
        output_triangles.push_back({
            tet_edge_vertices[tris[j + 2]], tet_edge_vertices[tris[j + 1]], tet_edge_vertices[tris[j]]
        });
      }
    }
  }

  // process prisms
  std::array<size_t, 9> prism_edge_vertices{};
  for (const auto& prism : prisms) {
    // index
    size_t index = 0;
    for (size_t j = 0; j < 6; ++j) {
      if (values[prism[5 - j]] > 0) {
        index |= 1 << j;
      }
    }

    const auto& edges = prism_table_edges[index];
    if (edges.empty()) {
      continue;
    }
    for (const auto& e : edges) {
      auto [v1, v2] = prism_edges[e];
      v1 = prism[v1];
      v2 = prism[v2];
      prism_edge_vertices[e] = get_new_vertex(v1, v2);
    }

    // create new triangles
    const bool oriented_prism = get_prism_orient(vertices3D[prism[0]], vertices3D[prism[1]],
                                                 vertices3D[prism[2]],
                                                 vertices3D[prism[3]], vertices3D[prism[4]],
                                                 vertices3D[prism[5]]) == 1;
    if (prism_insert_cycle_center) {
      const auto& cycles = prism_table_cycles[index];
      for (const auto& cycle : cycles) {
        if (oriented_prism) {
          if (cycle.size() == 3) {
            output_triangles.push_back({
                prism_edge_vertices[cycle[0]], prism_edge_vertices[cycle[1]],
                prism_edge_vertices[cycle[2]]
            });
          } else {
            // triangulate the cycle by inserting a new vertex at the center
            const size_t vid = output_vertices.size();
            auto& center = output_vertices.emplace_back();
            center = {0, 0, 0};
            for (size_t i = 0; i < cycle.size(); ++i) {
              center[0] += output_vertices[prism_edge_vertices[cycle[i]]][0];
              center[1] += output_vertices[prism_edge_vertices[cycle[i]]][1];
              center[2] += output_vertices[prism_edge_vertices[cycle[i]]][2];
              output_triangles.push_back({
                  prism_edge_vertices[cycle[i]], prism_edge_vertices[cycle[(i + 1) % cycle.size()]],
                  vid
              });
            }
            Float factor = 1.0 / static_cast<Float>(cycle.size());
            center[0] *= factor;
            center[1] *= factor;
            center[2] *= factor;
          }
        } else {
          if (cycle.size() == 3) {
            output_triangles.push_back({
                prism_edge_vertices[cycle[2]], prism_edge_vertices[cycle[1]],
                prism_edge_vertices[cycle[0]]
            });
          } else {
            // triangulate the cycle by inserting a new vertex at the center
            const size_t vid = output_vertices.size();
            auto& center = output_vertices.emplace_back();
            center = {0, 0, 0};
            for (size_t i = 0; i < cycle.size(); ++i) {
              center[0] += output_vertices[prism_edge_vertices[cycle[i]]][0];
              center[1] += output_vertices[prism_edge_vertices[cycle[i]]][1];
              center[2] += output_vertices[prism_edge_vertices[cycle[i]]][2];
              output_triangles.push_back({
                  prism_edge_vertices[cycle[(i + 1) % cycle.size()]], prism_edge_vertices[cycle[i]],
                  vid
              });
            }
            Float factor = 1.0 / static_cast<Float>(cycle.size());
            center[0] *= factor;
            center[1] *= factor;
            center[2] *= factor;
          }
        }
      }
    } else {
      const auto& tris = prism_table_tris[index];
      if (oriented_prism) {
        for (size_t j = 0; j < tris.size(); j += 3) {
          output_triangles.push_back({
              prism_edge_vertices[tris[j]], prism_edge_vertices[tris[j + 1]],
              prism_edge_vertices[tris[j + 2]]
          });
        }
      } else {
        for (size_t j = 0; j < tris.size(); j += 3) {
          output_triangles.push_back({
              prism_edge_vertices[tris[j + 2]], prism_edge_vertices[tris[j + 1]],
              prism_edge_vertices[tris[j]]
          });
        }
      }
    }
  }
}

template void MarchingTetPrism<double>(const std::vector<std::array<double, 4> >& vertices,
                                       const std::vector<std::array<size_t, 4> >& tets,
                                       const std::vector<std::array<size_t, 6> >& prisms,
                                       const std::vector<double>& values,
                                       const std::vector<std::array<double, 4> >& gradients,
                                       std::vector<std::array<double, 3> >& output_vertices,
                                       std::vector<std::array<size_t, 3> >& output_triangles,
                                       bool prism_insert_cycle_center);

template void MarchingTetPrism<float>(const std::vector<std::array<float, 4> >& vertices,
                                      const std::vector<std::array<size_t, 4> >& tets,
                                      const std::vector<std::array<size_t, 6> >& prisms,
                                      const std::vector<float>& values,
                                      const std::vector<std::array<float, 4> >& gradients,
                                      std::vector<std::array<float, 3> >& output_vertices,
                                      std::vector<std::array<size_t, 3> >& output_triangles,
                                      bool prism_insert_cycle_center);

template <typename Float>
void MarchingTet3D(const std::vector<std::array<Float, 3> >& vertices,
                   const std::vector<std::array<size_t, 4> >& tets,
                   const std::vector<Float>& values, const std::vector<std::array<Float, 3> >& gradients,
                   std::vector<std::array<Float, 3> >& output_vertices,
                   std::vector<std::array<size_t, 3> >& output_triangles) {
    assert(vertices.size() == values.size());
    const bool has_gradients = gradients.size() == vertices.size();
    output_vertices.clear();
    output_triangles.clear();
    
    // map edge to output vertex index
    std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> edge_map;
    
    // find if the edge is already processed, if not, create a new vertex
    auto get_new_vertex = [&edge_map, &output_vertices, &vertices, &values, &gradients, &has_gradients
    ](size_t v1, size_t v2) {
        if (v1 > v2) {
            std::swap(v1, v2);
        }
        auto edge = std::make_pair(v1, v2);
        if (edge_map.find(edge) == edge_map.end()) {
            edge_map[edge] = output_vertices.size();
            auto& new_vertex = output_vertices.emplace_back();
            if (has_gradients) {
                get_cubic_root(vertices[v1], vertices[v2], values[v1], values[v2], gradients[v1], gradients[v2],
                               new_vertex);
            } else {
                get_linear_root(vertices[v1], vertices[v2], values[v1], values[v2], new_vertex);
            }
            return output_vertices.size() - 1;
        } else {
            return edge_map[edge];
        }
    };
    
    // process tets
    std::array<size_t, 6> tet_edge_vertices{};
    for (const auto& tet : tets) {
        // index
        size_t index = 0;
        for (size_t j = 0; j < 4; ++j) {
            if (values[tet[3 - j]] > 0) {
                index |= 1 << j;
            }
        }
        
        const auto& edges = tet_table3D_edges[index];
        if (edges.empty()) {
            continue;
        }
        for (const auto& e : edges) {
            auto [v1, v2] = tet_edges3D[e];
            v1 = tet[v1];
            v2 = tet[v2];
            tet_edge_vertices[e] = get_new_vertex(v1, v2);
        }
        
        // create new triangles
        const auto& tris = tet_table3d_tris[index];
        if (get_tet_orient(vertices[tet[0]], vertices[tet[1]], vertices[tet[2]], vertices[tet[3]]) == 1) {
            for (size_t j = 0; j < tris.size(); j += 3) {
                output_triangles.push_back({
                    tet_edge_vertices[tris[j]], tet_edge_vertices[tris[j + 1]], tet_edge_vertices[tris[j + 2]]
                });
            }
        } else {
            for (size_t j = 0; j < tris.size(); j += 3) {
                output_triangles.push_back({
                    tet_edge_vertices[tris[j + 2]], tet_edge_vertices[tris[j + 1]], tet_edge_vertices[tris[j]]
                });
            }
        }
    }
}

template void MarchingTet3D<double>(const std::vector<std::array<double, 3> >& vertices,
                                    const std::vector<std::array<size_t, 4> >& tets,
                                    const std::vector<double>& values,
                                    const std::vector<std::array<double, 3> >& gradients,
                                    std::vector<std::array<double, 3> >& output_vertices,
                                    std::vector<std::array<size_t, 3> >& output_triangles);

template void MarchingTet3D<float>(const std::vector<std::array<float, 3> >& vertices,
                                   const std::vector<std::array<size_t, 4> >& tets,
                                   const std::vector<float>& values,
                                   const std::vector<std::array<float, 3> >& gradients,
                                   std::vector<std::array<float, 3> >& output_vertices,
                                   std::vector<std::array<size_t, 3> >& output_triangles);
} // namespace marching4D