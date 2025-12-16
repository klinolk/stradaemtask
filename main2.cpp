#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "utils/mesh.h"
#include "utils/LiteMath.h"
#include "utils/public_camera.h"
#include "utils/public_image.h"

#include <cstdio>
#include <cstring>
#include <SDL_keycode.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <SDL.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <memory>
#include <string>

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::int2;
using LiteMath::int3;
using LiteMath::int4;
using LiteMath::uint2;
using LiteMath::uint3;
using LiteMath::uint4;

// ============ ВОКСЕЛЬНЫЙ ИНТЕРФЕЙС ============
// Этот интерфейс позволит работать как с сеткой, так и с октодеревом

// 1. Структура вокселя с материалами
struct Voxel {
    uint32_t type;        // тип материала: 0=air, 1=grass, 2=dirt, 3=stone, 4=water
    uint32_t color;       // цвет в формате RGBA
    float3 normal;        // нормаль
    uint8_t density;      // плотность
    uint8_t metadata;     // дополнительные данные
    
    Voxel() : type(0), color(0xFF000000), normal(0,0,0), density(0), metadata(0) {}
    
    // Простой конструктор
    Voxel(uint32_t t, uint32_t c) : type(t), color(c), normal(0,0,0), density(0), metadata(0) {}
};

// 2. Абстрактный интерфейс для воксельного мира
class IVoxelWorld {
public:
    virtual ~IVoxelWorld() = default;
    
    // Основные методы доступа
    virtual Voxel getVoxel(int x, int y, int z) const = 0;
    virtual bool isSolid(int x, int y, int z) const = 0;
    virtual float3 getNormal(int x, int y, int z) const = 0;
    
    // Информация о размерах
    virtual int getSizeX() const = 0;
    virtual int getSizeY() const = 0;
    virtual int getSizeZ() const = 0;
    
    // Для отладки и оптимизации
    virtual size_t getMemoryUsage() const = 0;
    virtual std::string getDescription() const = 0;
    
    // Метод для трассировки лучей
    virtual bool rayCast(const float3& origin, const float3& direction,
                        float maxDist, float3& hitPos, float3& normal,
                        Voxel& hitVoxel) const = 0;
};

// 3. Реализация на основе регулярной сетки
class GridVoxelWorld : public IVoxelWorld {
private:
    std::vector<std::vector<std::vector<Voxel>>> grid;
    int sizeX, sizeY, sizeZ;
    
public:
    GridVoxelWorld(int sx, int sy, int sz) : sizeX(sx), sizeY(sy), sizeZ(sz) {
        grid.resize(sizeX);
        for (int x = 0; x < sizeX; x++) {
            grid[x].resize(sizeY);
            for (int y = 0; y < sizeY; y++) {
                grid[x][y].resize(sizeZ);
                // Инициализируем все как воздух
                for (int z = 0; z < sizeZ; z++) {
                    grid[x][y][z] = Voxel(0, 0xFF000000);
                }
            }
        }
    }
    
    // Установка вокселя (для генерации ландшафта)
    void setVoxel(int x, int y, int z, const Voxel& voxel) {
        if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ) {
            grid[x][y][z] = voxel;
        }
    }
    
    // Реализация интерфейса
    Voxel getVoxel(int x, int y, int z) const override {
        if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ) {
            return grid[x][y][z];
        }
        return Voxel(0, 0xFF000000); // Возвращаем воздух вне границ
    }
    
    bool isSolid(int x, int y, int z) const override {
        if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ) {
            return grid[x][y][z].type != 0; // 0 = воздух
        }
        return false;
    }
    
    float3 getNormal(int x, int y, int z) const override {
        // Вычисляем нормаль по соседям
        float3 normal(0.0f, 0.0f, 0.0f);
        
        if (x > 0 && !isSolid(x-1, y, z)) normal.x = -1.0f;
        else if (x < sizeX-1 && !isSolid(x+1, y, z)) normal.x = 1.0f;
        
        if (y > 0 && !isSolid(x, y-1, z)) normal.y = -1.0f;
        else if (y < sizeY-1 && !isSolid(x, y+1, z)) normal.y = 1.0f;
        
        if (z > 0 && !isSolid(x, y, z-1)) normal.z = -1.0f;
        else if (z < sizeZ-1 && !isSolid(x, y, z+1)) normal.z = 1.0f;
        
        if (LiteMath::length(normal) < 0.1f) {
            return float3(0.0f, 1.0f, 0.0f);
        }
        
        return LiteMath::normalize(normal);
    }
    
    bool rayCast(const float3& origin, const float3& direction,
                float maxDist, float3& hitPos, float3& normal,
                Voxel& hitVoxel) const override {
        
        // Преобразуем мировые координаты в координаты сетки
        float3 gridOrigin = origin + float3(sizeX/2.0f, 0, sizeZ/2.0f);
        float3 rayStart = gridOrigin;
        float3 dir = LiteMath::normalize(direction);
        float3 pos = rayStart;
        
        // Определяем текущий воксель
        int x = static_cast<int>(floor(pos.x));
        int y = static_cast<int>(floor(pos.y));
        int z = static_cast<int>(floor(pos.z));
        
        // Проверяем, находимся ли мы внутри сетки
        if (x < 0 || x >= sizeX || y < 0 || y >= sizeY || z < 0 || z >= sizeZ) {
            // Находим точку входа в сетку
            float tMin = 0.0f;
            float tMax = maxDist;
            
            for (int i = 0; i < 3; i++) {
                if (dir[i] != 0) {
                    float t1 = (0 - pos[i]) / dir[i];
                    float t2 = ((i == 0 ? sizeX : (i == 1 ? sizeY : sizeZ)) - 1 - pos[i]) / dir[i];
                    float tNear = std::min(t1, t2);
                    float tFar = std::max(t1, t2);
                    
                    tMin = std::max(tMin, tNear);
                    tMax = std::min(tMax, tFar);
                    
                    if (tMin > tMax) return false;
                }
            }
            
            if (tMin > 0) {
                pos = pos + dir * tMin;
                x = static_cast<int>(floor(pos.x));
                y = static_cast<int>(floor(pos.y));
                z = static_cast<int>(floor(pos.z));
            } else {
                return false;
            }
        }
        
        // Шаги по осям
        int stepX = (dir.x > 0) ? 1 : -1;
        int stepY = (dir.y > 0) ? 1 : -1;
        int stepZ = (dir.z > 0) ? 1 : -1;
        
        // Расстояние до следующей границы вокселя
        float nextX = (stepX > 0) ? (x + 1) : x;
        float nextY = (stepY > 0) ? (y + 1) : y;
        float nextZ = (stepZ > 0) ? (z + 1) : z;
        
        float tMaxX = (dir.x != 0) ? (nextX - pos.x) / dir.x : FLT_MAX;
        float tMaxY = (dir.y != 0) ? (nextY - pos.y) / dir.y : FLT_MAX;
        float tMaxZ = (dir.z != 0) ? (nextZ - pos.z) / dir.z : FLT_MAX;
        
        // Расстояние для перехода к следующему вокселю
        float tDeltaX = (dir.x != 0) ? fabs(1.0f / dir.x) : FLT_MAX;
        float tDeltaY = (dir.y != 0) ? fabs(1.0f / dir.y) : FLT_MAX;
        float tDeltaZ = (dir.z != 0) ? fabs(1.0f / dir.z) : FLT_MAX;
        
        float distance = 0;
        
        // Основной цикл DDA
        while (distance < maxDist) {
            // Проверяем границы
            if (x < 0 || x >= sizeX || y < 0 || y >= sizeY || z < 0 || z >= sizeZ) {
                break;
            }
            
            // Проверяем, попали ли в воксель поверхности
            if (isSolid(x, y, z)) {
                hitPos = pos - float3(sizeX/2.0f, 0, sizeZ/2.0f);
                hitVoxel = getVoxel(x, y, z);
                normal = getNormal(x, y, z);
                return true;
            }
            
            // Переход к следующему вокселю
            if (tMaxX < tMaxY && tMaxX < tMaxZ) {
                x += stepX;
                distance = tMaxX;
                tMaxX += tDeltaX;
            } else if (tMaxY < tMaxZ) {
                y += stepY;
                distance = tMaxY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                distance = tMaxZ;
                tMaxZ += tDeltaZ;
            }
            
            // Обновляем позицию
            pos = rayStart + dir * distance;
        }
        
        return false;
    }
    
    int getSizeX() const override { return sizeX; }
    int getSizeY() const override { return sizeY; }
    int getSizeZ() const override { return sizeZ; }
    
    size_t getMemoryUsage() const override {
        return sizeX * sizeY * sizeZ * sizeof(Voxel);
    }
    
    std::string getDescription() const override {
        return "Grid Voxel World (" + std::to_string(sizeX) + "x" + 
               std::to_string(sizeY) + "x" + std::to_string(sizeZ) + ")";
    }
};

// 4. Утилиты для материалов и цветов
namespace VoxelMaterials {
    // Цвета материалов (в формате ARGB)
    const uint32_t AIR_COLOR   = 0x00000000;
    const uint32_t GRASS_COLOR = 0xFF228B22;  // зеленый
    const uint32_t DIRT_COLOR  = 0xFF8B4513;  // коричневый
    const uint32_t STONE_COLOR = 0xFF808080;  // серый
    const uint32_t WATER_COLOR = 0xFF1E90FF;  // голубой
    
    Voxel createVoxel(uint32_t type, uint32_t color = 0xFFFFFFFF) {
        Voxel v;
        v.type = type;
        v.color = color;
        return v;
    }
    
    Voxel createAir() {
        return createVoxel(0, AIR_COLOR);
    }
    
    Voxel createGrass(float heightRatio = 1.0f) {
        // Немного варьируем цвет травы в зависимости от высоты
        uint8_t r = 34;  // 0x22
        uint8_t g = 139 + static_cast<uint8_t>((heightRatio - 0.5f) * 50); // 0x8B
        uint8_t b = 34;  // 0x22
        uint32_t color = 0xFF000000 | (r << 16) | (g << 8) | b;
        return createVoxel(1, color);
    }
    
    Voxel createDirt() {
        return createVoxel(2, DIRT_COLOR);
    }
    
    Voxel createStone() {
        return createVoxel(3, STONE_COLOR);
    }
    
    Voxel createWater() {
        Voxel v = createVoxel(4, WATER_COLOR);
        v.density = 100; // Вода имеет плотность
        return v;
    }
    
    float3 getColorAsFloat3(uint32_t color) {
        float r = ((color >> 16) & 0xFF) / 255.0f;
        float g = ((color >> 8) & 0xFF) / 255.0f;
        float b = (color & 0xFF) / 255.0f;
        return float3(r, g, b);
    }
}

// 5. Генератор ландшафта
namespace TerrainGenerator {
    void createHillyTerrain(GridVoxelWorld& world) {
        int sizeX = world.getSizeX();
        int sizeY = world.getSizeY();
        int sizeZ = world.getSizeZ();
        
        float baseHeight = sizeY * 0.3f;
        
        for (int x = 0; x < sizeX; x++) {
            for (int z = 0; z < sizeZ; z++) {
                // Периодическая функция для высоты
                float fx = sin(x * 0.1f) * 0.7f;
                float fz = cos(z * 0.08f) * 0.5f;
                float hills = sin(x * 0.03f + z * 0.05f) * 1.2f;
                float height = baseHeight + (fx + fz + hills) * 8.0f;
                
                int y_height = static_cast<int>(height);
                y_height = std::clamp(y_height, 0, sizeY - 1);
                
                // Заполняем столбец вокселей с разными материалами
                for (int y = 0; y <= y_height; y++) {
                    Voxel voxel;
                    float heightRatio = (float)y / sizeY;
                    
                    if (y == y_height) {
                        // Поверхность - трава
                        voxel = VoxelMaterials::createGrass(heightRatio);
                    } else if (y > y_height - 5) {
                        // Верхний слой - земля
                        voxel = VoxelMaterials::createDirt();
                    } else {
                        // Нижние слои - камень
                        voxel = VoxelMaterials::createStone();
                    }
                    
                    world.setVoxel(x, y, z, voxel);
                }
            }
        }
        
        // Добавляем озеро в центре
        int centerX = sizeX / 2;
        int centerZ = sizeZ / 2;
        int lakeRadius = 15;
        
        for (int x = centerX - lakeRadius; x <= centerX + lakeRadius; x++) {
            for (int z = centerZ - lakeRadius; z <= centerZ + lakeRadius; z++) {
                float dx = x - centerX;
                float dz = z - centerZ;
                float dist = sqrt(dx*dx + dz*dz);
                
                if (dist <= lakeRadius) {
                    // Убираем землю под озером
                    for (int y = 0; y < 10; y++) {
                        world.setVoxel(x, y, z, VoxelMaterials::createAir());
                    }
                    // Добавляем воду
                    for (int y = 10; y < 12; y++) {
                        if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ) {
                            world.setVoxel(x, y, z, VoxelMaterials::createWater());
                        }
                    }
                }
            }
        }
    }
    
    void createFlatTerrain(GridVoxelWorld& world, float height = 20.0f) {
        // Простая плоская местность для тестирования
        int sizeX = world.getSizeX();
        int sizeY = world.getSizeY();
        int sizeZ = world.getSizeZ();
        
        int y_height = static_cast<int>(height);
        y_height = std::clamp(y_height, 0, sizeY - 1);
        
        for (int x = 0; x < sizeX; x++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y <= y_height; y++) {
                    Voxel voxel;
                    if (y == y_height) {
                        voxel = VoxelMaterials::createGrass(0.5f);
                    } else {
                        voxel = VoxelMaterials::createDirt();
                    }
                    world.setVoxel(x, y, z, voxel);
                }
            }
        }
    }
}

// ============ ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ============
std::unique_ptr<IVoxelWorld> g_voxelWorld;

// ============ РАЗРЕШЕНИЕ ЭКРАНА ============
static constexpr int SCREEN_WIDTH  = 640;
static constexpr int SCREEN_HEIGHT = 480;

// ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============
float rad_to_deg(float rad) { return rad * 180.0f / LiteMath::M_PI; }

uint32_t float3_to_RGBA8(float3 c) {
    uint8_t r = (uint8_t)(std::clamp(c.x, 0.0f, 1.0f) * 255.0f);
    uint8_t g = (uint8_t)(std::clamp(c.y, 0.0f, 1.0f) * 255.0f);
    uint8_t b = (uint8_t)(std::clamp(c.z, 0.0f, 1.0f) * 255.0f);
    return 0xFF000000 | (r << 16) | (g << 8) | b;
}

// ============ РЕНДЕРИНГ ============
void renderVoxelWorld(const Camera& camera, const IVoxelWorld& world, 
                     uint32_t* out_image, int W, int H) {
    
    LiteMath::float4x4 view = LiteMath::lookAt(camera.pos, camera.target, camera.up);
    LiteMath::float4x4 proj = LiteMath::perspectiveMatrix(
        rad_to_deg(camera.fov_rad), (float)W/(float)H, camera.z_near, camera.z_far);
    LiteMath::float4x4 viewProjInv = LiteMath::inverse4x4(proj * view);
    
    const float3 light_dir = LiteMath::normalize(float3(-1.0f, -1.0f, -1.0f));
    
    // Убираем антиалиасинг для скорости
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            // Получаем луч через пиксель
            float u = (x + 0.5f) / W;
            float v = (y + 0.5f) / H;
            float ndc_x = 2.0f * u - 1.0f;
            float ndc_y = 1.0f - 2.0f * v; // Инвертируем ось Y
            
            float4 point_NDC = float4(ndc_x, ndc_y, 0.0f, 1.0f);
            float4 point_W = viewProjInv * point_NDC;
            float3 point = LiteMath::to_float3(point_W) / point_W.w;
            float3 ray_pos = camera.pos;
            float3 ray_dir = LiteMath::normalize(point - ray_pos);
            
            // Трассируем луч через мир
            float3 hitPos, normal;
            Voxel hitVoxel;
            float3 color(0.0f, 0.0f, 0.0f);
            
            if (world.rayCast(ray_pos, ray_dir, 1000.0f, hitPos, normal, hitVoxel)) {
                // Базовое освещение Ламберта
                float lambert = std::max(0.0f, LiteMath::dot(normal, -light_dir));
                
                // Получаем цвет из вокселя
                float3 base_color = VoxelMaterials::getColorAsFloat3(hitVoxel.color);
                color = base_color * (0.25f + 0.75f * lambert);
            }
            
            out_image[y * W + x] = float3_to_RGBA8(color);
        }
    }
}

void draw_frame_example(const Camera& camera, std::vector<uint32_t>& pixels) {
    if (g_voxelWorld) {
        renderVoxelWorld(camera, *g_voxelWorld, pixels.data(), SCREEN_WIDTH, SCREEN_HEIGHT);
    }
}

// ============ УПРАВЛЕНИЕ КАМЕРОЙ ============
struct FreeCameraModel {
    enum class CameraMoveType : uint8_t {
        NONE,
        TUMBLE,
        TRACK,
        DOLLY
    };
    FreeCameraModel() = default;
    CameraMoveType move_type{CameraMoveType::NONE};
    int2 mouse_pos;
    float theta{0};
    float phi{0};
    float3 look_at{0, 0, 0};
    float dist_to_target{10};
};

FreeCameraModel freecam_model;
constexpr float kRotateAmpl = 0.005f;
constexpr float kPanAmpl = 0.01f;
constexpr float kScrollAmpl = 0.1f;

enum EventFlags {
    EF_NONE = 0,
    EF_SHIFT_DOWN = 1 << 0,
    EF_CONTROL_DOWN = 1 << 1,
    EF_ALT_DOWN = 1 << 2,
    EF_LEFT_DOWN = 1 << 3,
    EF_MIDDLE_DOWN = 1 << 4,
    EF_RIGHT_DOWN = 1 << 5
};

void OnMousePressed(int flags, int2 location) {
    freecam_model.mouse_pos = location;
    if (flags & EF_ALT_DOWN) {
        freecam_model.move_type = (flags & EF_LEFT_DOWN)     ? FreeCameraModel::CameraMoveType::TUMBLE
                                  : (flags & EF_MIDDLE_DOWN) ? FreeCameraModel::CameraMoveType::TRACK
                                  : (flags & EF_RIGHT_DOWN)  ? FreeCameraModel::CameraMoveType::DOLLY
                                                             : FreeCameraModel::CameraMoveType::NONE;
    }
}

void OnMouseReleased() {
    freecam_model.move_type = FreeCameraModel::CameraMoveType::NONE;
}

void OnMouseMoved(int flags, int2 location, Camera& camera) {
    if (freecam_model.move_type == FreeCameraModel::CameraMoveType::NONE) {
        return;
    }

    int2 delta = location - freecam_model.mouse_pos;
    freecam_model.mouse_pos = location;

    switch (freecam_model.move_type) {
    case FreeCameraModel::CameraMoveType::TUMBLE: {
        freecam_model.theta -= delta.x * kRotateAmpl;
        freecam_model.phi -= delta.y * kRotateAmpl;
        freecam_model.phi = std::clamp(freecam_model.phi, -LiteMath::M_PI / 2 + 0.1f, LiteMath::M_PI / 2 - 0.1f);
        float x = freecam_model.dist_to_target * cos(freecam_model.phi) * sin(freecam_model.theta);
        float y = freecam_model.dist_to_target * sin(freecam_model.phi);
        float z = freecam_model.dist_to_target * cos(freecam_model.phi) * cos(freecam_model.theta);
        camera.pos = freecam_model.look_at + float3(x, y, z);
        camera.target = freecam_model.look_at;
        break;
    }
    case FreeCameraModel::CameraMoveType::TRACK: {
        float3 forward = LiteMath::normalize(camera.target - camera.pos);
        float3 right = LiteMath::normalize(LiteMath::cross(float3(0, 1, 0), forward));
        float3 up = LiteMath::normalize(LiteMath::cross(forward, right));

        float3 move = right * (-delta.x * kPanAmpl) + up * (-delta.y * kPanAmpl);
        camera.pos += move;
        camera.target += move;
        freecam_model.look_at += move;
        break;
    }
    case FreeCameraModel::CameraMoveType::DOLLY: {
        float3 forward = LiteMath::normalize(camera.target - camera.pos);
        float3 move = forward * ((delta.x + delta.y) * kScrollAmpl);
        camera.pos += move;
        camera.target += move;
        freecam_model.dist_to_target = LiteMath::length(camera.pos - freecam_model.look_at);
        break;
    }
    default:
        break;
    }
}

void OnMouseWheel(int delta, Camera& camera) {
    float3 forward = LiteMath::normalize(camera.target - camera.pos);
    float3 move = forward * (delta * kScrollAmpl);
    camera.pos += move;
    camera.target += move;
    freecam_model.dist_to_target = LiteMath::length(camera.pos - freecam_model.look_at);
}

void WASD(Camera& camera, float dt) {
    float moveSpeed = 10.0f * dt;
    float3 forward = LiteMath::normalize(camera.target - camera.pos);
    float3 right = LiteMath::normalize(LiteMath::cross(float3(0, 1, 0), forward));
    float3 up = LiteMath::normalize(LiteMath::cross(forward, right));

    const Uint8* keystate = SDL_GetKeyboardState(NULL);
    float3 move(0, 0, 0);
    if (keystate[SDL_SCANCODE_W]) {
        move += forward * moveSpeed;
    }
    if (keystate[SDL_SCANCODE_S]) {
        move -= forward * moveSpeed;
    }
    if (keystate[SDL_SCANCODE_A]) {
        move += right * moveSpeed;
    }
    if (keystate[SDL_SCANCODE_D]) {
        move -= right * moveSpeed;
    }
    if (keystate[SDL_SCANCODE_Q]) {
        move += up * moveSpeed;
    }
    if (keystate[SDL_SCANCODE_E]) {
        move -= up * moveSpeed;
    }

    camera.pos += move;
    camera.target += move;
    freecam_model.look_at += move;

    freecam_model.dist_to_target = LiteMath::length(camera.pos - freecam_model.look_at);
    float3 dir = LiteMath::normalize(camera.pos - camera.target);
    freecam_model.theta = atan2(dir.x, dir.z);
    freecam_model.phi = asin(dir.y);
}

void InitializeFreeCameraFromCamera(const Camera& camera) {
    freecam_model.look_at = camera.target;
    freecam_model.dist_to_target = LiteMath::length(camera.pos - camera.target);

    float3 dir = LiteMath::normalize(camera.pos - camera.target);
    freecam_model.theta = atan2(dir.x, dir.z);
    freecam_model.phi = asin(dir.y);
}

// ============ ОСНОВНАЯ ФУНКЦИЯ ============
int main(int argc, char** args) {
    printf("=== Воксельный рендерер с интерфейсом ===\n");
    printf("Готов к интеграции с октодеревом!\n\n");
    
    // 1. Создаем воксельный мир (пока регулярная сетка)
    const int WORLD_SIZE_X = 128;
    const int WORLD_SIZE_Y = 64;
    const int WORLD_SIZE_Z = 128;
    
    printf("Создание сетки %dx%dx%d...\n", WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z);
    auto gridWorld = std::make_unique<GridVoxelWorld>(WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z);
    
    // 2. Заполняем мир тестовым ландшафтом
    printf("Генерация холмистого ландшафта...\n");
    TerrainGenerator::createHillyTerrain(*gridWorld);
    
    // 3. Сохраняем указатель на интерфейс
    g_voxelWorld = std::move(gridWorld);
    
    printf("Ландшафт сгенерирован.\n");
    printf("Описание: %s\n", g_voxelWorld->getDescription().c_str());
    printf("Используемая память: %.2f MB\n\n", 
           g_voxelWorld->getMemoryUsage() / (1024.0f * 1024.0f));
    
    // 4. Инициализация SDL
    std::vector<uint32_t> pixels(SCREEN_WIDTH * SCREEN_HEIGHT, 0xFFFFFFFF);

    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
        std::cerr << "Error initializing SDL: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Voxel World (готов для октодерева)", 
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          SCREEN_WIDTH, SCREEN_HEIGHT, 
                                          SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    if (!window) {
        std::cerr << "Error creating window: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        SCREEN_WIDTH,
        SCREEN_HEIGHT);

    if (!texture) {
        std::cerr << "Texture could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Event ev;
    bool running = true;

    Camera camera;
    // Настройка камеры для обзора ландшафта
    camera.pos = float3(0.0f, 50.0f, 100.0f);
    camera.target = float3(0.0f, 20.0f, 0.0f);
    camera.up = float3(0.0f, 1.0f, 0.0f);
    camera.fov_rad = LiteMath::M_PI / 4.0f; // 45 градусов
    camera.z_near = 1.0f;
    camera.z_far = 300.0f;

    InitializeFreeCameraFromCamera(camera);
    
    bool alt_pressed = false;
    bool mouse_left = false;
    bool mouse_middle = false;
    bool mouse_right = false;

    auto time = std::chrono::high_resolution_clock::now();
    auto prev_time = time;
    float time_from_start = 0;
    uint32_t frameNum = 0;

    printf("Управление:\n");
    printf("  - Alt + ЛКМ: Вращение камеры\n");
    printf("  - Alt + СКМ: Панорамирование\n");
    printf("  - Alt + ПКМ: Движение вперед/назад\n");
    printf("  - Колесико: Зум\n");
    printf("  - WASD: Движение камеры\n");
    printf("  - Q/E: Движение вверх/вниз\n");
    printf("  - ESC: Выход\n\n");

    // Основной цикл
    while (running) {
        // Обработка ввода
        while (SDL_PollEvent(&ev) != 0) {
            switch (ev.type) {
            case SDL_QUIT:
                running = false;
                break;
            
            case SDL_KEYDOWN:
                if (ev.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }
                // Обновление модификаторов
                if (ev.key.keysym.sym == SDLK_LALT || ev.key.keysym.sym == SDLK_RALT) {
                    alt_pressed = true;
                }
                break;
            
            case SDL_KEYUP:
                if (ev.key.keysym.sym == SDLK_LALT || ev.key.keysym.sym == SDLK_RALT) {
                    alt_pressed = false;
                }
                break;
            
            case SDL_MOUSEBUTTONDOWN:
                if (ev.button.button == SDL_BUTTON_LEFT) {
                    mouse_left = true;
                }
                if (ev.button.button == SDL_BUTTON_MIDDLE) {
                    mouse_middle = true;
                }
                if (ev.button.button == SDL_BUTTON_RIGHT) {
                    mouse_right = true;
                }

                if (alt_pressed) {
                    int flags = EF_ALT_DOWN;
                    if (mouse_left) {
                        flags |= EF_LEFT_DOWN;
                    }
                    if (mouse_middle) {
                        flags |= EF_MIDDLE_DOWN;
                    }
                    if (mouse_right) {
                        flags |= EF_RIGHT_DOWN;
                    }

                    OnMousePressed(flags, LiteMath::int2(ev.button.x, ev.button.y));
                }
                break;
            
            case SDL_MOUSEBUTTONUP:
                if (ev.button.button == SDL_BUTTON_LEFT) {
                    mouse_left = false;
                }
                if (ev.button.button == SDL_BUTTON_MIDDLE) {
                    mouse_middle = false;
                }
                if (ev.button.button == SDL_BUTTON_RIGHT) {
                    mouse_right = false;
                }
                OnMouseReleased();
                break;
            
            case SDL_MOUSEMOTION:
                if (alt_pressed && (mouse_left || mouse_middle || mouse_right)) {
                    int flags = EF_ALT_DOWN;
                    if (mouse_left) {
                        flags |= EF_LEFT_DOWN;
                    }
                    if (mouse_middle) {
                        flags |= EF_MIDDLE_DOWN;
                    }
                    if (mouse_right) {
                        flags |= EF_RIGHT_DOWN;
                    }

                    OnMouseMoved(flags, LiteMath::int2(ev.motion.x, ev.motion.y), camera);
                }
                break;
            
            case SDL_MOUSEWHEEL:
                OnMouseWheel(ev.wheel.y, camera);
                break;
            }
        }

        // Обновление времени
        prev_time = time;
        time = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float, std::milli>(time - prev_time).count() / 1000.0f;
        time_from_start += dt;
        frameNum++;

        if (frameNum % 60 == 0) {
            printf("FPS: %.1f\n", 1.0f / dt);
        }

        // Движение камеры
        WASD(camera, dt);

        // Рендеринг сцены
        draw_frame_example(camera, pixels);

        // Обновление текстуры
        SDL_UpdateTexture(texture, nullptr, pixels.data(), SCREEN_WIDTH * sizeof(uint32_t));

        // Очистка и отрисовка
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        
        // Ограничение FPS
        SDL_Delay(1);
    }

    // Очистка
    SDL_DestroyWindow(window);
    SDL_Quit();

    printf("\nПрограмма завершена. Готово для интеграции с октодеревом!\n");
    return 0;
}