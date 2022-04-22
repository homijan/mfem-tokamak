#ifndef MAGNETICFIELDTSCHIRNHAUSSEN_HPP
#define MAGNETICFIELDTSCHIRNHAUSSEN_HPP

#include "mfem.hpp"

using namespace mfem;

class MagneticFieldTschirnhaussen
{
public:
  MagneticFieldTschirnhaussen(double B0_=1, double theta_=3*M_PI/4, 
    double scale_=1.0) : B0(B0_), theta(theta_), scale(scale_) 
  { 
    // Tschirnhaussen domain is a 2x2 square at angle 45deg, which requires
    // sqrt(2) scaling of x and y with respect to 1x1 domain. 
    scale *= sqrt(2.0); 
  }

  void setB(const Vector &x, Vector &B) const
  {
    // k = dy / dx
    double k = kTangent(x);
    // bhat = [bx, by], (bx^2 + by^2)^0.5 = 1, k = by / bx 
    // => by = k * bx => bx^2 + (k * bx)^2 = 1 => bx = (1 / 1 + kx^2)^0.5
    B(0) = sqrt(1.0 / (1.0 + k*k));
    B(1) = k * B(0);
    // Scale be with respect to the flux surface? 
    B *= B0;
  }

  double fluxSurface(const Vector &x) const
  {
    // Get reference coordinates of Tschirnhaussen curve.
    Vector xRef(2);
    xReference(x, xRef);
    // Evaluate flux coordinate at given point.
    return fluxSurfaceTschirnhaussen(xRef);
  }

private:
  double B0;
  double theta;
  double scale;

  void xReference(const Vector &x, Vector &xRef) const
  {
    // The x-point is shifted to the origin
    xRef(0) = scale * (cos(theta) * x(0) - sin(theta) * x(1)) + 0.5;
    xRef(1) = scale * (sin(theta) * x(0) + cos(theta) * x(1));
  }
  // Get k slope of the Tschirnhausen tangent at xReference
  double kReference(const Vector &xRef) const
  {
    // Tschirnhaussen function y = (x^3 + x^2 + d)^0.5
    // and dydx = 0.5 * (3 * x^2 + 2 * x) / (x^3 + x^2 + d)^0.5
    double x = xRef(0);
    double y = abs(xRef(1));
    double d = fluxSurfaceTschirnhaussen(xRef);
    double epsilon = 1e-16;
    double kRef = 0.5 * (3.0 * x * x + 2.0 * x) / 
      (sqrt(x*x*x + x*x + d) + epsilon);
    if (xRef(1) < 0.0) kRef *= -1.0;
    return kRef;
  }
  double kTangent(const Vector &x) const
  {
    Vector xRef(2);
    xReference(x, xRef);
    double kRef = kReference(xRef);
    return (kRef * cos(-theta) + sin(-theta)) / 
      (cos(-theta) - kRef * sin(-theta));
  }
  double fluxSurfaceTschirnhaussen(const Vector &xReference) const
  {
    double x = xReference(0);
    double y = xReference(1);
    // Tschirnhaussen curve y^2 - x^3 - x^2 = d, where d is the coordinate of 
    // flux surface
    return y*y - x*x*x - x*x;
  }
};

double TschirnhaussenSource_function(const Vector &x)
{
  // Tokamak center in ref coords is [-0.66675, 0], where ref coords are 
  // shifted by 0.5 in x. The square in ref has diagonal 2.0 long.
  // -1.16675 in "un-shifted" ref coords
  double distance = 1.16675 * sqrt(2.0) / 2.0;
  double xSource = distance * cos(M_PI / 4.0);
  double ySource = distance * sin(M_PI / 4.0);
  double rx=x(0)-xSource, ry=x(1)-ySource;

  return exp(-4*(rx*rx+ry*ry));
}

#endif
