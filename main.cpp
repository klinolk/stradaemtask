#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "utils/mesh.h"
#include "utils/LiteMath.h"
#include "utils/public_camera.h"
#include "utils/public_image.h"
#include "utils/voxel.h"
#include "utils/octree.h"

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

// ============ РЕАЛИЗАЦИЯ GRID VOXEL WORLD ============
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
                for (int z = 0; z < sizeZ; z++) {
                    grid[x][y][z] = Voxel(0, 0xFF000000);
                }
            }
        }
    }
    
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
        return Voxel(0, 0xFF000000);
    }
    
    bool isSolid(int x, int y, int z) const override {
        if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ) {
            return grid[x][y][z].type != 0;
        }
        return false;
    }
    
    float3 getNormal(int x, int y, int z) const override {
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
        
        // Упрощенная реализация для тестирования
        float3 gridOrigin = origin + float3(sizeX/2.0f, 0, sizeZ/2.0f);
        float3 rayStart = gridOrigin;
        float3 dir = LiteMath::normalize(direction);
        
        float t = 0;
        float step = 0.5f;
        
        while (t < maxDist) {
            float3 pos = rayStart + dir * t;
            int x = static_cast<int>(floor(pos.x));
            int y = static_cast<int>(floor(pos.y));
            int z = static_cast<int>(floor(pos.z));
            
            if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ) {
                if (isSolid(x, y, z)) {
                    hitPos = pos - float3(sizeX/2.0f, 0, sizeZ/2.0f);
                    hitVoxel = getVoxel(x, y, z);
                    normal = getNormal(x, y, z);
                    return true;
                }
            }
            
            t += step;
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
        return "Grid (" + std::to_string(sizeX) + "x" + 
               std::to_string(sizeY) + "x" + std::to_string(sizeZ) + ")";
    }
};

// ============ УТИЛИТЫ ДЛЯ МАТЕРИАЛОВ ============
namespace VoxelMaterials {
    const uint32_t AIR_COLOR   = 0x00000000;
    const uint32_t GRASS_COLOR = 0xFF228B22;
    const uint32_t DIRT_COLOR  = 0xFF8B4513;
    const uint32_t STONE_COLOR = 0xFF808080;
    const uint32_t WATER_COLOR = 0xFF1E90FF;
    
    Voxel createVoxel(uint32_t type, uint32_t color = 0xFFFFFFFF) {
        Voxel v;
        v.type = type;
        v.color = color;
        return v;
    }
    
    Voxel createAir() { return createVoxel(0, AIR_COLOR); }
    Voxel createGrass(float heightRatio = 1.0f) { return createVoxel(1, GRASS_COLOR); }
    Voxel createDirt() { return createVoxel(2, DIRT_COLOR); }
    Voxel createStone() { return createVoxel(3, STONE_COLOR); }
    Voxel createWater() { 
        Voxel v = createVoxel(4, WATER_COLOR);
        v.density = 100;
        return v;
    }
    
    float3 getColorAsFloat3(uint32_t color) {
        float r = ((color >> 16) & 0xFF) / 255.0f;
        float g = ((color >> 8) & 0xFF) / 255.0f;
        float b = (color & 0xFF) / 255.0f;
        return float3(r, g, b);
    }
}

// ============ ГЕНЕРАТОР ЛАНДШАФТА ============
namespace TerrainGenerator {
    void createSimpleTerrain(GridVoxelWorld& world) {
        int sizeX = world.getSizeX();
        int sizeY = world.getSizeY();
        int sizeZ = world.getSizeZ();
        
        for (int x = 0; x < sizeX; x++) {
            for (int z = 0; z < sizeZ; z++) {
                float height = 10.0f + sin(x * 0.1f) * 2.0f + cos(z * 0.1f) * 2.0f;
                int y_height = static_cast<int>(height);
                y_height = std::clamp(y_height, 0, sizeY - 1);
                
                for (int y = 0; y <= y_height; y++) {
                    Voxel voxel;
                    if (y == y_height) {
                        voxel = VoxelMaterials::createGrass();
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
std::unique_ptr<GridVoxelWorld> g_gridWorld;
std::unique_ptr<OctreeVoxelWorld> g_octreeWorld;
bool g_useOctree = true;

// ============ КОНВЕРТЕР ============
std::unique_ptr<OctreeVoxelWorld> convertGridToOctree(const GridVoxelWorld& gridWorld) {
    int sizeX = gridWorld.getSizeX();
    int sizeY = gridWorld.getSizeY();
    int sizeZ = gridWorld.getSizeZ();
    
    printf("Converting Grid -> Octree (%dx%dx%d)...\n", sizeX, sizeY, sizeZ);
    
    auto octreeWorld = std::make_unique<OctreeVoxelWorld>(sizeX, sizeY, sizeZ);
    
    int voxelCount = 0;
    for (int x = 0; x < sizeX; x++) {
        for (int y = 0; y < sizeY; y++) {
            for (int z = 0; z < sizeZ; z++) {
                Voxel v = gridWorld.getVoxel(x, y, z);
                if (v.type != 0) {
                    octreeWorld->setVoxel(x, y, z, v);
                    voxelCount++;
                }
            }
        }
    }
    
    printf("  Inserted %d voxels\n", voxelCount);
    octreeWorld->compressTree();
    printf("  Conversion complete.\n");
    
    return octreeWorld;
}

// ============ РАЗРЕШЕНИЕ ЭКРАНА ============
static constexpr int SCREEN_WIDTH  = 640;
static constexpr int SCREEN_HEIGHT = 480;

// ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============
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
        45.0f, (float)W/(float)H, 0.1f, 300.0f);
    LiteMath::float4x4 viewProjInv = LiteMath::inverse4x4(proj * view);
    
    const float3 light_dir = LiteMath::normalize(float3(-1.0f, -1.0f, -1.0f));
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float u = (x + 0.5f) / W;
            float v = (y + 0.5f) / H;
            float ndc_x = 2.0f * u - 1.0f;
            float ndc_y = 1.0f - 2.0f * v;
            
            float4 point_NDC = float4(ndc_x, ndc_y, 0.0f, 1.0f);
            float4 point_W = viewProjInv * point_NDC;
            float3 point = LiteMath::to_float3(point_W) / point_W.w;
            float3 ray_pos = camera.pos;
            float3 ray_dir = LiteMath::normalize(point - ray_pos);
            
            float3 hitPos, normal;
            Voxel hitVoxel;
            float3 color(0.0f, 0.0f, 0.0f);
            
            if (world.rayCast(ray_pos, ray_dir, 1000.0f, hitPos, normal, hitVoxel)) {
                float lambert = std::max(0.0f, LiteMath::dot(normal, -light_dir));
                float3 base_color = VoxelMaterials::getColorAsFloat3(hitVoxel.color);
                color = base_color * (0.25f + 0.75f * lambert);
            }
            
            out_image[y * W + x] = float3_to_RGBA8(color);
        }
    }
}

// ============ УПРАВЛЕНИЕ КАМЕРОЙ ============
Camera camera;

void WASD(Camera& camera, float dt) {
    float moveSpeed = 10.0f * dt;
    float3 forward = LiteMath::normalize(camera.target - camera.pos);
    float3 right = LiteMath::normalize(LiteMath::cross(float3(0, 1, 0), forward));
    float3 up = LiteMath::normalize(LiteMath::cross(forward, right));

    const Uint8* keystate = SDL_GetKeyboardState(NULL);
    float3 move(0, 0, 0);
    if (keystate[SDL_SCANCODE_W]) move += forward * moveSpeed;
    if (keystate[SDL_SCANCODE_S]) move -= forward * moveSpeed;
    if (keystate[SDL_SCANCODE_A]) move += right * moveSpeed;
    if (keystate[SDL_SCANCODE_D]) move -= right * moveSpeed;
    if (keystate[SDL_SCANCODE_Q]) move += up * moveSpeed;
    if (keystate[SDL_SCANCODE_E]) move -= up * moveSpeed;

    camera.pos += move;
    camera.target += move;
}

// ============ ОСНОВНАЯ ФУНКЦИЯ ============
int main() {
    printf("=== Voxel Renderer with Octree ===\n");
    
    // Создаем мир
    const int WORLD_SIZE_X = 32;
    const int WORLD_SIZE_Y = 16;
    const int WORLD_SIZE_Z = 32;
    
    printf("Creating grid %dx%dx%d...\n", WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z);
    g_gridWorld = std::make_unique<GridVoxelWorld>(WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z);
    
    printf("Generating terrain...\n");
    TerrainGenerator::createSimpleTerrain(*g_gridWorld);
    
    printf("Converting to octree...\n");
    g_octreeWorld = convertGridToOctree(*g_gridWorld);
    
    g_voxelWorld.reset(g_octreeWorld.get());
    g_useOctree = true;
    
    printf("Current world: %s\n", g_voxelWorld->getDescription().c_str());
    printf("Memory usage: %.2f MB\n\n", g_voxelWorld->getMemoryUsage() / (1024.0f * 1024.0f));
    
    // Инициализация SDL
    std::vector<uint32_t> pixels(SCREEN_WIDTH * SCREEN_HEIGHT, 0xFFFFFFFF);

    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
        std::cerr << "Error initializing SDL: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Voxel Octree", 
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          SCREEN_WIDTH, SCREEN_HEIGHT, 
                                          SDL_WINDOW_SHOWN);

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

    // Настройка камеры
    camera.pos = float3(0.0f, 15.0f, 30.0f);
    camera.target = float3(0.0f, 5.0f, 0.0f);
    camera.up = float3(0.0f, 1.0f, 0.0f);
    camera.fov_rad = LiteMath::M_PI / 4.0f;
    camera.z_near = 0.1f;
    camera.z_far = 300.0f;

    auto prev_time = std::chrono::high_resolution_clock::now();

    printf("Controls:\n");
    printf("  - O: Toggle between grid and octree\n");
    printf("  - WASD: Move camera\n");
    printf("  - ESC: Exit\n\n");

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
                else if (ev.key.keysym.sym == SDLK_o) {
                    g_useOctree = !g_useOctree;
                    
                    if (g_useOctree) {
                        printf("Switching to octree...\n");
                        g_octreeWorld = convertGridToOctree(*g_gridWorld);
                        g_voxelWorld.reset(g_octreeWorld.get());
                    } else {
                        printf("Switching to grid...\n");
                        g_voxelWorld.reset(g_gridWorld.get());
                    }
                    
                    printf("Current world: %s\n", g_voxelWorld->getDescription().c_str());
                    printf("Memory usage: %.2f MB\n", 
                           g_voxelWorld->getMemoryUsage() / (1024.0f * 1024.0f));
                }
                break;
            }
        }

        // Обновление времени
        auto current_time = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float, std::milli>(current_time - prev_time).count() / 1000.0f;
        prev_time = current_time;

        // Движение камеры
        WASD(camera, dt);

        // Рендеринг
        renderVoxelWorld(camera, *g_voxelWorld, pixels.data(), SCREEN_WIDTH, SCREEN_HEIGHT);
        
        // Обновление текстуры
        SDL_UpdateTexture(texture, nullptr, pixels.data(), SCREEN_WIDTH * sizeof(uint32_t));
        
        // Очистка и отрисовка
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        
        SDL_Delay(16);
    }

    // Очистка
    SDL_DestroyWindow(window);
    SDL_Quit();

    printf("\nProgram finished.\n");
    return 0;
}