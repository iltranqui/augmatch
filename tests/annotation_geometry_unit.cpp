#include "augmatch/augmatch.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
  using namespace augmatch;
  const Geometry g{4, 3, 1, 0, 0, 4, 3, true, false};
  const PointXY polygon_points[]{{0,0},{3,0},{3,2},{0,2}};
  const std::size_t offsets[]{0,4};
  const PolygonView polygon{polygon_points,4,offsets,1};
  PointXY poly_out[4]{};
  transform_polygon(polygon, poly_out, g);
  if (poly_out[0].x != 3 || poly_out[1].x != 0 || poly_out[2].x != 0 || poly_out[3].x != 3) return 1;
  const PointXY path[]{{0,1},{1,1},{3,1}};
  PointXY path_out[3]{};
  transform_line_string({path,3}, path_out, g);
  if (path_out[0].x != 3 || path_out[1].x != 2 || path_out[2].x != 0) return 2;
  bool rejected = false;
  try { validate_line_string({path,1},4,3); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) return 3;

  const float heat_in[]{0,1,2,3,4,5,6,7,8,9,10,11};
  float heat_out[12]{};
  HeatmapView src{heat_in,4,3,1,sizeof(float),4*sizeof(float),12*sizeof(float)};
  MutableHeatmapView dst{heat_out,4,3,1,sizeof(float),4*sizeof(float),12*sizeof(float)};
  transform_heatmap(src,dst,g);
  for (int y=0;y<3;++y) for(int x=0;x<4;++x)
    if (heat_out[y*4+x] != heat_in[y*4+(3-x)]) return 4;
  // Applying the same flip twice restores coordinates and continuous samples.
  transform_heatmap({heat_out,4,3,1,sizeof(float),4*sizeof(float),12*sizeof(float)},dst,g);
  for (int i=0;i<12;++i) if (heat_out[i] != heat_in[i]) return 5;
  return 0;
}
