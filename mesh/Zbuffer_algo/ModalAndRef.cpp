#include "3DMM.h"

void ZBuffer(double* vertex, double* tri, double* texture, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind)
{
	int i,j;
	int x,y;

	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];
	double* h = new double[ntri];
	double* imgh = new double[width * height];
	double* tritex = new double[ntri * nChannels];

	for(i = 0; i < width * height; i++)
	{
		imgh[i] = -99999999999999;
		tri_ind[i] = -1;
	}

	for(i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3*i]);
		int p2 = int(tri[3*i+1]);
		int p3 = int(tri[3*i+2]);

		point1[2*i] = vertex[3*p1];	point1[2*i+1] = vertex[3*p1+1];
		point2[2*i] = vertex[3*p2];	point2[2*i+1] = vertex[3*p2+1];
		point3[2*i] = vertex[3*p3];	point3[2*i+1] = vertex[3*p3+1];

		double cent3d_z = (vertex[3*p1+2] + vertex[3*p2+2] + vertex[3*p3+2]) / 3;

		h[i] = cent3d_z;

		for(j = 0; j < nChannels; j++)
		{
			tritex[nChannels*i+j] = (texture[nChannels*p1+j] + texture[nChannels*p2+j] + texture[nChannels*p3+j]) / 3;
		}
	}

    point p, pt1, pt2, pt3;

	//init image
	for(i = 0; i < width * height * nChannels; i++)
	{
		img[i] = src_img[i];
	}

	for(i = 0; i < ntri; i++)
	{
		pt1.x = point1[2*i]; pt1.y= point1[2*i+1];
		pt2.x = point2[2*i]; pt2.y = point2[2*i+1];
		pt3.x = point3[2*i]; pt3.y = point3[2*i+1];

		int x_min = (int)ceil((double)min(min(pt1.x, pt2.x), pt3.x));
		int x_max = (int)floor((double)max(max(pt1.x, pt2.x), pt3.x));

		int y_min = (int)ceil((double)min(min(pt1.y, pt2.y), pt3.y));
		int y_max = (int)floor((double)max(max(pt1.y, pt2.y), pt3.y));

		if(x_max < x_min || y_max < y_min || x_max > width-1 || x_min < 0 || y_max > height-1 || y_min < 0)
			continue;
		
		for(y = y_min; y <= y_max; y++)
		{
			for (x = x_min; x <= x_max; x++)
			{
				p.x = (double)x;
				p.y = (double)y;
				if( imgh[y * width + x] < h[i] && PointInTri(p, pt1, pt2, pt3))
				{
					imgh[y * width + x] = h[i];
					for(j = 0; j < nChannels; j++)
					{
						img[j * width * height + y * width + x] =  tritex[nChannels * i + j];
					}
					tri_ind[y * width + x] = i;
				}
			}
		}
	}


	delete[] point1;
	delete[] point2;
	delete[] point3;
	delete[] h;
	delete[] imgh;
	delete[] tritex;
}

void ZBufferTri(double* vertex, double* tri, double* texture_tri, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind)
{
	int i,j;
	int x,y;

	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];
	double* h = new double[ntri];
	double* imgh = new double[width * height];
	double* tritex = texture_tri;

	for(i = 0; i < width * height; i++)
	{
		imgh[i] = -99999999999999;
		tri_ind[i] = -1;
	}

	for(i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3*i]);
		int p2 = int(tri[3*i+1]);
		int p3 = int(tri[3*i+2]);

		point1[2*i] = vertex[3*p1];	point1[2*i+1] = vertex[3*p1+1];
		point2[2*i] = vertex[3*p2];	point2[2*i+1] = vertex[3*p2+1];
		point3[2*i] = vertex[3*p3];	point3[2*i+1] = vertex[3*p3+1];

		double cent3d_z = (vertex[3*p1+2] + vertex[3*p2+2] + vertex[3*p3+2]) / 3;

		h[i] = cent3d_z;
	}

	point p, pt1, pt2, pt3;

	//init image
	for(i = 0; i < width * height * nChannels; i++)
	{
		img[i] = src_img[i];
	}

	for(i = 0; i < ntri; i++)
	{
		pt1.x = point1[2*i]; pt1.y= point1[2*i+1];
		pt2.x = point2[2*i]; pt2.y = point2[2*i+1];
		pt3.x = point3[2*i]; pt3.y = point3[2*i+1];
		
		int x_min = (int)ceil((double)min(min(pt1.x, pt2.x), pt3.x));
		int x_max = (int)floor((double)max(max(pt1.x, pt2.x), pt3.x));

		int y_min = (int)ceil((double)min(min(pt1.y, pt2.y), pt3.y));
		int y_max = (int)floor((double)max(max(pt1.y, pt2.y), pt3.y));

		if(x_max < x_min || y_max < y_min)
			continue;
		
		x_min = min(max(x_min, 0), width-1);
		x_max = min(max(x_max, 0), width-1);
		y_min = min(max(y_min, 0), height-1);
		y_max = min(max(y_max, 0), height-1);
		
		for(y = y_min; y <= y_max; y++)
		{
			for (x = x_min; x <= x_max; x++)
			{
				p.x = (double)x;
				p.y = (double)y;
				if( imgh[y * width + x] < h[i] && PointInTri(p, pt1, pt2, pt3))
				{
					imgh[y * width + x] = h[i];
					for(j = 0; j < nChannels; j++)
					{
						img[j * width * height + y * width + x] =  tritex[nChannels * i + j];
					}
					tri_ind[y * width + x] = i;
				}
			}
		}
	}


	delete[] point1;
	delete[] point2;
	delete[] point3;
	delete[] h;
	delete[] imgh;
}

bool PointInTri(point p, point pt1, point pt2, point pt3)
{
	double pointx = p.x;
	double pointy = p.y;

	double pt1x = pt1.x;
	double pt1y = pt1.y;

	double pt2x = pt2.x;
	double pt2y = pt2.y;

	double pt3x = pt3.x;
	double pt3y = pt3.y;

	double v0x = pt3x - pt1x;
	double v0y = pt3y - pt1y;

	double v1x = pt2x - pt1x;
	double v1y = pt2y - pt1y;

	double v2x = pointx - pt1x;
	double v2y = pointy - pt1y;

	double dot00 = v0x * v0x + v0y * v0y;
	double dot01 = v0x * v1x + v0y * v1y;
	double dot02 = v0x * v2x + v0y * v2y;
	double dot11 = v1x * v1x + v1y * v1y;
	double dot12 = v1x * v2x + v1y * v2y;

	double inverDeno = 0;
	if((dot00 * dot11 - dot01 * dot01) == 0)
		inverDeno = 0;
	else
		inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

	double u = (dot11 * dot02 - dot01 * dot12) * inverDeno;

	if(u < 0 || u > 1)
		return 0;

	double v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

	if(v < 0 || v > 1)
		return 0;

	return u + v <= 1;
}