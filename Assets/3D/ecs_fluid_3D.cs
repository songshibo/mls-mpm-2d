using Unity.Mathematics;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Burst;
using System.Runtime.InteropServices;
using UnityEngine.Profiling;

public class ecs_fluid_3D : MonoBehaviour
{
    struct Particle
    {
        public float3 pos;
        public float3 v;
        public float3x3 C;
        public float mass;
        public float padding;
    };

    struct Cell
    {
        public float3 v;
        public float mass;
        public float padding;
    };

    //* Simulation parameters
    const float dt = 0.01f;
    const int iterations = (int)(1 / dt);
    const float gravity = -0.2f;
    //* Grid properties
    const int grid_res = 64;
    const int num_cell = grid_res * grid_res * grid_res;
    NativeArray<Cell> grid;
    //* Particle properties
    int num_particle = 0;
    NativeArray<Particle> ps;
    //* Initialization parameters
    float spacing = 0.5f;
    int num_edge = 8;
    float init_mass = 0.5f;
    //* fluid parameters
    const float rest_density = 4.0f;
    const float dynamic_viscosity = 0.1f;
    //* equation of state
    const float eos_stiffness = 10.0f;
    const float eos_power = 4;
    //* Render Setting
    [SerializeField]
    GameObject particle_prefab = null;
    List<Transform> prefab_list = null;
    TransformAccessArray ps_array;
    float render_range = 1.0f;

    void Start()
    {
        prefab_list = new List<Transform>();
        Initialize();
        ps_array = new TransformAccessArray(prefab_list.ToArray());
    }

    void Initialize()
    {
        List<float3> init_pos = new List<float3>();
        float3 center = math.float3(grid_res / 2.0f, grid_res / 2.0f, grid_res / 2.0f);
        for (float j = center.y - num_edge / 2; j < center.y + num_edge / 2; j += spacing)
        {
            for (float i = center.x - num_edge / 2; i < center.x + num_edge / 2; i += spacing)
            {
                for (float k = center.z - num_edge / 2; k < center.z + num_edge / 2; k += spacing)
                {
                    init_pos.Add(math.float3(i, j, k));
                }
            }
        }

        num_particle = init_pos.Count;
        ps = new NativeArray<Particle>(num_particle, Allocator.Persistent);

        for(int i = 0; i < num_particle; ++i)
        {
            Particle p = new Particle();
            p.pos = init_pos[i];
            p.v = 0;
            p.C = 0;
            p.mass = init_mass;
            ps[i] = p;

            float3 worldPos = (p.pos - grid_res/2.0f) / grid_res * render_range;
            GameObject tmp = Instantiate(particle_prefab, new Vector3(worldPos.x, worldPos.y, worldPos.z), Quaternion.identity) as GameObject;
            tmp.name = i.ToString();
            prefab_list.Add(tmp.GetComponent<Transform>());
        }

        grid = new NativeArray<Cell>(num_cell, Allocator.Persistent);
        for (int i = 0; i < num_cell; ++i)
        {
            Cell c = new Cell();
            c.v = 0;
            grid[i] = c;
        }
    }

    void Update()
    {
        for (int i = 0; i < iterations; ++i)
        {
            Simulation();
        }

        // if (Input.GetKey(KeyCode.DownArrow))
        // {
        // ClearGrid_single();
        // P2G_first_single();
        // P2G_second_single();
        // UpdateGrid_single();
        // G2P_single();
        // PrefabRenderer_single();
        // }
    }

    void Simulation()
    {
        Profiler.BeginSample("Clear Grid");
        new ClearGrid()
        {
            grid = grid
        }.Schedule(num_cell, 64).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("P2G first parallel");
        new P2G_first()
        {
            grid = grid,
            ps = ps,
        }.Schedule(num_particle, (int)(num_edge / spacing)).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("P2G second parallel");
        new P2G_second()
        {
            grid = grid,
            ps = ps,
        }.Schedule(num_particle, (int)(num_edge / spacing)).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("Update Grid");
        new UpdateGrid()
        {
            grid = grid
        }.Schedule(num_cell, 64).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("G2P");
        JobHandle G2P = new G2P()
        {
            grid = grid,
            ps = ps,
        }.Schedule(num_particle, (int)(num_edge / spacing));
        Profiler.EndSample();

        Profiler.BeginSample("Prefab Render");
        new PrefabRenderer()
        {
            ps = ps,
            render_range = render_range,
            grid_res = grid_res
        }.Schedule(ps_array, G2P).Complete();
        Profiler.EndSample();
    }

    [BurstCompile]
    struct ClearGrid : IJobParallelFor
    {
        public NativeArray<Cell> grid;
        public void Execute(int i)
        {
            Cell c = grid[i];
            c.v = 0;
            c.mass = 0;
            grid[i] = c;
        }
    }

    [BurstCompile]
    unsafe struct P2G_first : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Cell> grid;
        [ReadOnly]
        public NativeArray<Particle> ps;
        public void Execute(int i)
        {
            float3* w = stackalloc float3[3];
            Particle p = ps[i];

            uint3 cell_idx = (uint3)p.pos;
            float3 cell_diff = (p.pos - cell_idx) - 0.5f;

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            for(uint y = 0; y < 3; ++y)
            {
                for(uint x = 0; x < 3; ++x)
                {
                    for(uint z = 0; z < 3; ++z)
                    {
                        float weight = w[x].x * w[y].y * w[z].z;

                        uint3 current_cell_idx = math.uint3(cell_idx.x + x - 1, cell_idx.y + y - 1, cell_idx.z + z - 1);
                        float3 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                        float3 Q = math.mul(p.C, cell_dist);

                        float mass_contrib = weight * p.mass;

                        int index_1d = (int)current_cell_idx.y * grid_res * grid_res + (int)current_cell_idx.x * grid_res + (int)current_cell_idx.z;
                        Cell c = grid[index_1d];

                        c.mass += mass_contrib;
                        c.v += mass_contrib * (p.v + Q);

                        grid[index_1d] = c;
                    }
                }
            }
        }
    }

    [BurstCompile]
    unsafe struct P2G_second : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Cell> grid;
        [ReadOnly]
        public NativeArray<Particle> ps;
        public void Execute(int i)
        {
            float3* w = stackalloc float3[3];
            Particle p = ps[i];

            uint3 cell_idx = (uint3)p.pos;
            float3 cell_diff = (p.pos - cell_idx) - 0.5f;

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float density = 0;
            
            for(uint y = 0; y < 3; ++y)
            {
                for(uint x = 0; x < 3; ++x)
                {
                    for(uint z = 0; z < 3; ++z)
                    {
                        float weight = w[x].x * w[y].y * w[z].z;
                        int index_1d = (int)(cell_idx.y + y - 1) * grid_res * grid_res + (int)(cell_idx.x + x - 1) * grid_res + (int)(cell_idx.z + z - 1);
                        density += grid[index_1d].mass * weight;
                    }
                }
            }
            float volume = p.mass / density;

            float pressure = math.max(-0.1f, eos_stiffness * (math.pow(density / rest_density, eos_power) - 1));
            float3x3 stress = math.float3x3(
                -pressure, 0, 0,
                0, -pressure, 0,
                0, 0, -pressure
            );

            float3x3 vel_grad = p.C;
            float3x3 vel_grad_T = math.transpose(vel_grad);
            float3x3 strain = vel_grad + vel_grad_T;

            float3x3 visc_term = dynamic_viscosity * strain;
            stress += visc_term;

            for(uint y = 0; y < 3; ++y)
            {
                for(uint x = 0; x < 3; ++x)
                {
                    for(uint z = 0; z < 3; ++z)
                    {
                        float weight = w[x].x * w[y].y * w[z].z;

                        uint3 current_cell_idx = math.uint3(cell_idx.x + x - 1, cell_idx.y + y - 1, cell_idx.z + z - 1);
                        float3 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                        int index_1d = (int)current_cell_idx.y * grid_res * grid_res + (int)current_cell_idx.x * grid_res + (int)current_cell_idx.z;
                        Cell c = grid[index_1d];

                        float3 momentum = math.mul(-volume * 4 * stress * dt, cell_dist);
                        c.v += momentum;
                        grid[index_1d] = c;
                    }
                }
            }
        }
    }

    [BurstCompile]
    struct UpdateGrid : IJobParallelFor
    {
        public NativeArray<Cell> grid;
        public void Execute(int i)
        {
            Cell c = grid[i];
            if(c.mass > 0)
            {
                c.v /= c.mass;
                c.v += dt * math.float3(0, gravity, 0);

                int z = i / (grid_res * grid_res);
                int x = (i - z * (grid_res * grid_res)) / grid_res;
                int y = (i - z * (grid_res * grid_res)) % grid_res;
                c.v.z = (z < 2)||(z > grid_res -3)?0:c.v.z;
                c.v.x = (x < 2)||(x > grid_res -3)?0:c.v.x;
                c.v.y = (y < 2)||(y > grid_res -3)?0:c.v.y;

                grid[i] = c;
            }
        }
    }

    [BurstCompile]
    unsafe struct G2P : IJobParallelFor
    {
        [ReadOnly]public NativeArray<Cell> grid;
        public NativeArray<Particle> ps;
        public void Execute(int i)
        {
            Particle p = ps[i];
            p.v = 0;

            float3* w = stackalloc float3[3];

            uint3 cell_idx = (uint3)p.pos;
            float3 cell_diff = (p.pos - cell_idx) - 0.5f;

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float3x3 B = 0;

            for(uint y = 0; y < 3; ++y)
            {
                for(uint x = 0; x < 3; ++x)
                {
                    for(uint z = 0; z < 3; ++z)
                    {
                        float weight = w[x].x * w[y].y * w[z].z;

                        uint3 current_cell_idx = math.uint3(cell_idx.x + x - 1, cell_idx.y + y - 1, cell_idx.z + z - 1);
                        float3 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                        int index_1d = (int)current_cell_idx.y * grid_res * grid_res + (int)current_cell_idx.x * grid_res + (int)current_cell_idx.z;
                        
                        float3 weighted_vel = grid[index_1d].v * weight;
                        float3x3 B_term = math.float3x3(weighted_vel.x * cell_dist.x, weighted_vel.y * cell_dist.y, weighted_vel.z * cell_dist.z);
                        B += B_term;
                        p.v += weighted_vel;
                    }
                }
            }
            p.C = B * 4;
            p.pos += p.v * dt;
            p.pos = math.clamp(p.pos, 1, grid_res - 2);

            float3 x_n = p.pos + p.v;
            float wall_min = 3;
            float wall_max = (float)grid_res - 4;
            p.v.x += (x_n.x < wall_min)? wall_min - x_n.x: 0;
            p.v.x += (x_n.x > wall_max)? wall_max - x_n.x: 0;
            p.v.y += (x_n.y < wall_min)? wall_min - x_n.y: 0;
            p.v.y += (x_n.y > wall_max)? wall_max - x_n.y: 0;
            p.v.z += (x_n.z < wall_min)? wall_min - x_n.z: 0;
            p.v.z += (x_n.z > wall_max)? wall_max - x_n.z: 0;

            ps[i] = p;
        }
    }

    [BurstCompile]
    unsafe struct PrefabRenderer : IJobParallelForTransform
    {
        [ReadOnly]
        public NativeArray<Particle> ps;
        [ReadOnly]
        public float render_range;
        [ReadOnly]
        public int grid_res;

        public void Execute(int i, TransformAccess transform)
        {
            Particle p = ps[i];
            transform.position = (p.pos - grid_res/2.0f)/grid_res * render_range;
        }
    }

    void ClearGrid_single()
    {
        for (int i = 0; i < num_cell; ++i)
        {
            Cell c = grid[i];
            c.v = 0;
            c.mass = 0;
            grid[i] = c;
        }
    }

    void P2G_first_single()
    {
        for(int i = 0; i < num_particle; ++i)
        {
            float3[] w = new float3[3];
            Particle p = ps[i];

            uint3 cell_idx = (uint3)p.pos;
            float3 cell_diff = (p.pos - cell_idx) - 0.5f;

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            for (uint y = 0; y < 3; ++y)
            {
                for (uint x = 0; x < 3; ++x)
                {
                    for (uint z = 0; z < 3; ++z)
                    {
                        float weight = w[x].x * w[y].y * w[z].z;

                        uint3 current_cell_idx = math.uint3(cell_idx.x + x - 1, cell_idx.y + y - 1, cell_idx.z + z - 1);
                        float3 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                        float3 Q = math.mul(p.C, cell_dist);

                        float mass_contrib = weight * p.mass;

                        int index_1d = (int)current_cell_idx.y * grid_res * grid_res + (int)current_cell_idx.x * grid_res + (int)current_cell_idx.z;
                        Cell c = grid[index_1d];

                        c.mass += mass_contrib;
                        c.v += mass_contrib * (p.v + Q);

                        grid[index_1d] = c;
                    }
                }
            }
        }
    }

    void P2G_second_single()
    {
        for(int i = 0; i < num_particle; ++i)
        {
            float3[] w = new float3[3];
            Particle p = ps[i];

            uint3 cell_idx = (uint3)p.pos;
            float3 cell_diff = (p.pos - cell_idx) - 0.5f;

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float density = 0;

            for (uint y = 0; y < 3; ++y)
            {
                for (uint x = 0; x < 3; ++x)
                {
                    for (uint z = 0; z < 3; ++z)
                    {
                        float weight = w[x].x * w[y].y * w[z].z;
                        int index_1d = (int)(cell_idx.y + y - 1) * grid_res * grid_res + (int)(cell_idx.x + x - 1) * grid_res + (int)(cell_idx.z + z - 1);
                        density += grid[index_1d].mass * weight;
                    }
                }
            }
            float volume = p.mass / density;

            float pressure = math.max(-0.1f, eos_stiffness * (math.pow(density / rest_density, eos_power) - 1));
            float3x3 stress = math.float3x3(
                -pressure, 0, 0,
                0, -pressure, 0,
                0, 0, -pressure
            );

            float3x3 vel_grad = p.C;
            float3x3 vel_grad_T = math.transpose(vel_grad);
            float3x3 strain = vel_grad + vel_grad_T;

            float3x3 visc_term = dynamic_viscosity * strain;
            stress += visc_term;

            for (uint y = 0; y < 3; ++y)
            {
                for (uint x = 0; x < 3; ++x)
                {
                    for (uint z = 0; z < 3; ++z)
                    {
                        float weight = w[x].x * w[y].y * w[z].z;

                        uint3 current_cell_idx = math.uint3(cell_idx.x + x - 1, cell_idx.y + y - 1, cell_idx.z + z - 1);
                        float3 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                        int index_1d = (int)current_cell_idx.y * grid_res * grid_res + (int)current_cell_idx.x * grid_res + (int)current_cell_idx.z;
                        Cell c = grid[index_1d];

                        float3 momentum = math.mul(-volume * 4 * stress * dt, cell_dist);
                        c.v += momentum;
                        grid[index_1d] = c;
                    }
                }
            }
        }
    }

    void UpdateGrid_single()
    {
        for (int i = 0; i < num_cell; ++i)
        {
            Cell c = grid[i];
            if (c.mass > 0)
            {
                c.v /= c.mass;
                c.v += dt * math.float3(0, gravity, 0);

                // int y = i / (grid_res * grid_res);
                // int x = (i - y * (grid_res * grid_res)) / grid_res;
                // int z = (i - y * (grid_res * grid_res) - x * grid_res);
                // c.v.z = (z < 2) || (z > grid_res - 3) ? 0 : c.v.z;
                // c.v.x = (x < 2) || (x > grid_res - 3) ? 0 : c.v.x;
                // c.v.y = (y < 2) || (y > grid_res - 3) ? 0 : c.v.y;

                grid[i] = c;
            }
        } 
    }

    void G2P_single()
    {
        for(int i = 0; i < num_particle; ++i)
        {
            Particle p = ps[i];
            p.v = 0;

            float3[] w = new float3[3];

            uint3 cell_idx = (uint3)p.pos;
            float3 cell_diff = (p.pos - cell_idx) - 0.5f;

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float3x3 B = 0;

            for (uint y = 0; y < 3; ++y)
            {
                for (uint x = 0; x < 3; ++x)
                {
                    for (uint z = 0; z < 3; ++z)
                    {
                        float weight = w[x].x * w[y].y * w[z].z;

                        uint3 current_cell_idx = math.uint3(cell_idx.x + x - 1, cell_idx.y + y - 1, cell_idx.z + z - 1);
                        float3 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                        int index_1d = (int)current_cell_idx.y * grid_res * grid_res + (int)current_cell_idx.x * grid_res + (int)current_cell_idx.z;

                        float3 weighted_vel = grid[index_1d].v * weight;
                        float3x3 B_term = math.float3x3(weighted_vel.x * cell_dist.x, weighted_vel.y * cell_dist.y, weighted_vel.z * cell_dist.z);
                        B += B_term;
                        p.v += weighted_vel;
                    }
                }
            }
            p.C = B * 4;
            p.pos += p.v * dt;
            p.pos = math.clamp(p.pos, 1, grid_res - 2);

            float3 x_n = p.pos + p.v;
            float wall_min = 3 - 1;
            float wall_max = (float)grid_res - 4 + 1;
            p.v.x += (x_n.x < wall_min) ? wall_min - x_n.x : 0;
            p.v.x += (x_n.x > wall_max) ? wall_max - x_n.x : 0;
            p.v.y += (x_n.y < wall_min) ? wall_min - x_n.y : 0;
            p.v.y += (x_n.y > wall_max) ? wall_max - x_n.y : 0;
            p.v.z += (x_n.z < wall_min) ? wall_min - x_n.z : 0;
            p.v.z += (x_n.z > wall_max) ? wall_max - x_n.z : 0;

            ps[i] = p;
        }
    }

    void PrefabRenderer_single()
    {
        for(int i = 0; i < num_particle; ++i)
        {
            float3 pos = (ps[i].pos - grid_res/2.0f) / grid_res * render_range;
            prefab_list[i].position = new Vector3(pos.x, pos.y, pos.z);
        }
    }
    void OnDestroy()
    {
        ps.Dispose();
        grid.Dispose();
        ps_array.Dispose();
    }
}
