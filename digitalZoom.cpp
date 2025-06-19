#include <iostream>
struct hiDev_Rect
{
    int x;
    int y;
    int width;
    int height;
};

void setDigitalZoom(int videosource, double zoomFactor, int offsetX, int offsetY)
{
    int ret = 0;
    hiDev_Rect m_rect;
    int srcWidth = 0;
    int srcHeight = 0;

    if (videosource == 0) {
        srcWidth = 1920;
        srcHeight = 1080;
    } else if (videosource == 1) {
        srcWidth = 640;
        srcHeight = 512;
    }
    // 计算缩放后的 ROI（取景框）
    int centerX = srcWidth / 2 + offsetX;  // 变倍中心 X
    int centerY = srcHeight / 2 + offsetY; // 变倍中心 Y
    int cropWidth = static_cast<int>(srcWidth / zoomFactor);
    int cropHeight = static_cast<int>(srcHeight / zoomFactor);

    int x1 = std::max(centerX - cropWidth / 2, 0);
    int y1 = std::max(centerY - cropHeight / 2, 0);
    int x2 = std::min(x1 + cropWidth, srcWidth);
    int y2 = std::min(y1 + cropHeight, srcHeight);

    m_rect = {x1, y1, x2 - x1, y2 - y1};

    std::cout << "m_rect.x: " << m_rect.x << "\nm_rect.y: " << m_rect.y 
    << "\nm_rect.width: " << m_rect.width << "\nm_rect.height: " << m_rect.height << std::endl; 
    if (videosource == 0) {
        // ret = hiDev_setViChnAbsCrop(0, 0, true, m_rect);
    } else if (videosource == 1) {
        // ret = hiDev_setViChnAbsCrop(3, 0, true, m_rect);
        // ret = hiDev_setViChnAbsCrop(3, 0, true, {320, 104, 640, 512});
    } else {
    }
}
int main() {
    setDigitalZoom(0, 2, 0, 0);
    return 0;

}