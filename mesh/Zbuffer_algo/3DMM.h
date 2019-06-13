#ifndef MM3D_H
#define MM3D_H

#include <math.h>
#include "3DMMGlobal.h"
#include <algorithm>

using namespace std;

#define INF 1E20

class point
{
 public:
    double x;
    double y;

    float dot(point p)
    {
        return this->x * p.x + this->y * p.y;
    }

    point operator-(const point& p)
    {
        point np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    }

    point operator+(const point& p)
    {
        point np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    }

    point operator*(double s)
    {
        point np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    }
};

// 3DMM and Reference Frame Mapping
void ZBuffer(double* vertex, double* tri, double* texture, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind);
void ZBufferTri(double* vertex, double* tri, double* texture_tri, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind);

bool PointInTri(point p, point pt1, point pt2, point pt3);

#endif