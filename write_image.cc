#include <iostream>
#include <random>


#define STB_DEFINE                                                     
#include <stb.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <Eigen/Dense>

#include <igl/readOBJ.h>
#include <igl/barycentric_coordinates.h>

using namespace std;
using namespace Eigen;

typedef Matrix<double, 3, 2> Triangle2d;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::uniform_int_distribution<> dis(1, 255);

RowVector3d baryCoords(const Triangle2d &T, RowVector3d P) {
    RowVector3d f1 = {T(2,0) - T(0, 0), T(1, 0) - T(0, 0), T(0, 0) - P[0]};
    RowVector3d f2 = {T(2,1) - T(0, 1), T(1, 1) - T(0, 1), T(0, 1) - P[1]};
    RowVector3d f = f1.cross(f2);
    if (abs(f[2] < 1)) return RowVector3d(-1, 1, 1);
    return {1. - (f[0] + f[1]) / f[2], f[1]/f[2], f[0]/f[2]};
}

struct Image {
    int H, W, NUM_CHANNELS;
    uint8_t *pixels;
    MatrixXd z_buffer;
    Image(int H_, int W_, int NUM_CHANNELS_) : H(H_), W(W_), NUM_CHANNELS(NUM_CHANNELS_) {
        pixels = new uint8_t[W * H * NUM_CHANNELS];
        z_buffer.resize(H, W); z_buffer.setZero();

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                setColor(i, j, {115, 55, 77});
            }
        }
    }

    void set_wireframce(const MatrixXd &V, const MatrixXi &F) {
        for (int i = 0; i < F.rows(); i++) {
            for (int j = 0; j < 3; j++) {
                RowVector3d v1 = V.row(F(i,j)), v2 = V.row(F(i, (j+1) %3));
                int x1 = W - (v1[0] + 1.)*W/2., y1 = H - (v1[1] + 1.)*H/2.;
                int x2 = W - (v2[0] + 1.)*W/2., y2 = H - (v2[1] + 1.)*H/2.;
                line({y1, x1}, {y2, x2}, {255, 255, 255});
            }
        }
    }

    void set_obj(const MatrixXd &V, const MatrixXi &F) {
        for (int i = 0; i < F.rows(); i++) {
            Triangle2d T;
            for (int j = 0; j < 3; j++) {
                RowVector3d v1 = V.row(F(i,j)), v2 = V.row(F(i, (j+1) %3));
                int x1 = W - (v1[0] + 1.)*W/2., y1 = H - (v1[1] + 1.)*H/2.;
                int x2 = W - (v2[0] + 1.)*W/2., y2 = H - (v2[1] + 1.)*H/2.;
                T.row(j) = RowVector2d(y1, x1);
                //line({y1, x1}, {y2, x2}, {255, 255, 255});
            }
            RowVector3d f1 = V.row(F(i, 1)) - V.row(F(i, 0));
            RowVector3d f2 = V.row(F(i, 1)) - V.row(F(i, 2));

            RowVector3d n = f1.cross(f2).normalized();
            RowVector3d light_dir = {0,0, -1};
            double intensity = n.dot(light_dir);
           // RowVector3i rand_color = {dis(gen), dis(gen), dis(gen)};
            if (intensity > 0) {
                RowVector3i color = {255.*intensity, 255.*intensity, 255.*intensity};
                triangle(T, color);
            }
        }
    }

    void setColor(const int i, const int j, const RowVector3i &color) {
        assert(i < H && j < W);
        assert(color.maxCoeff() <= 255);
        pixels[i*W*NUM_CHANNELS + j*3 + 0] = color[0];
        pixels[i*W*NUM_CHANNELS + j*3 + 1] = color[1];
        pixels[i*W*NUM_CHANNELS + j*3 + 2] = color[2];
    }

    void line(RowVector2d p1, RowVector2d p2, RowVector3i color) {
        bool steep = false;
        if (abs(p1[0] - p2[0]) < abs(p1[1] - p2[1])) {
            swap(p1[0], p1[1]);
            swap(p2[0], p2[1]);
            steep = true;
        } 
        if (p1[0] > p2[0]) {
            swap(p1[0], p2[0]);
            swap(p1[1], p2[1]);
        }

        int dx = p2[0] - p1[0]; 
        int dy = p2[1] - p1[1];
        float derror = abs(dy)*2;
        float error = 0;
        int y = p1[1];

        for (int x = p1[0]; x <= p2[0]; x++) {
            if (steep) {
                setColor(y, x, color);
            } else {
                setColor(x, y, color);
            }
            error += derror;
            if (error > dx) {
                y += (p2[1] > p1[1] ? 1 : -1);
                error -= dx*2;
            }
        }
    }

    void triangle(Triangle2d &T, RowVector3i color) {
        RowVector2d bbmin = T.colwise().minCoeff();
        RowVector2d bbmax = T.colwise().maxCoeff();
        bbmin[0] = max(0., bbmin[0]), bbmin[1] = max(0., bbmin[1]);
        bbmax[0] = min(H-1., bbmax[0]), bbmax[1] = min(W-1., bbmax[1]);

        RowVector3d p;
        for (p[0] = bbmin[0]; p[0] < bbmax[0]; p[0]++) {
            for (p[1] = bbmin[1]; p[1] < bbmax[1]; p[1]++) {
                RowVector3d bCords = baryCoords(T, p);
                if (bCords.minCoeff() < 0) continue;
                
                p[2] = T.colwise().sum()[2]*bCords[2];

                if (true || z_buffer((int)p[0], (int)p[1]) <= p[2]) {
                    z_buffer((int)p[0], (int)p[1]) = p[2];
                    setColor(p[0], p[1], color);
                }
                

            }
        }

    }


    void toTga() {
        stbi_write_tga("test.tga", W, H , NUM_CHANNELS, pixels);
    }
};




int main() {
    int NUM_CHANNELS = 3, H = 560, W = 1200;

    MatrixXd V; MatrixXi F;
    igl::readOBJ("../armadillo.obj", V, F);

    Image *img = new Image(H, W, NUM_CHANNELS);
    //img->set_wireframce(V, F);
    img->set_obj(V, F);

    img->toTga();
    cout << "hellowlimowli" << endl;
}