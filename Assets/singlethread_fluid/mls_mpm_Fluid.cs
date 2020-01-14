using Unity.Mathematics;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Burst;
using System.Runtime.InteropServices;
using UnityEngine.Profiling;

public class mls_mpm_Fluid : MonoBehaviour
{
    struct Particle
    {
        public float2 pos;
        public float2 v;
        public float2x2 C;
        public float mass;
        public float padding;
    };

    struct Cell
    {
        public float2 v;
        public float mass;
        public float padding;
    };

    const int grid_res = 128;
    const int num_cell = grid_res * grid_res;

    const float dt = 0.2f;
    const int iterations = (int)(1 / dt);
    const float gravity = -0.25f;
    // fluid parameters
    const float rest_density = 4.0f;
    const float dynamic_viscosity = 0.1f;
    // equation of state
    const float eos_stiffness = 10.0f;
    const float eos_power = 4;

    NativeArray<Particle> ps;
    NativeArray<Cell> grid;
    int num_particle = 0;

    [SerializeField]
    float spacing = 0.5f;
    [SerializeField]
    int init_edge = 8;
    [SerializeField]
    bool useJobs = true;

    [Space]
    [SerializeField]
    [ReadOnlyWhenPlaying]
    bool usePrefab = true;
    [SerializeField]
    float render_range = 6.0f;
    [SerializeField]
    GameObject particle_prefab = null;
    List<Transform> prefab_list = null;
    TransformAccessArray particle_array;

    SimRenderer sim_renderer;

    void Start()
    {
        prefab_list = usePrefab? new List<Transform>() : null;
        Initialize();
        if(usePrefab)
        {
            particle_array = new TransformAccessArray(prefab_list.ToArray());
        }
        else
        {
            sim_renderer = GameObject.FindObjectOfType<SimRenderer>();
            sim_renderer.Initialise(num_particle, Marshal.SizeOf(new Particle()));
        }
    }

    void Initialize()
    {
        List<float2> init_pos = new List<float2>();
        float2 center = math.float2(grid_res / 2.0f, grid_res / 2.0f);
        for (float i = center.x - init_edge / 2; i < center.x + init_edge / 2; i += spacing)
        {
            for (float j = center.y - init_edge / 2; j < center.y + init_edge / 2; j += spacing)
            {
                init_pos.Add(math.float2(i, j));
            }
        }

        num_particle = init_pos.Count;
        ps = new NativeArray<Particle>(num_particle, Allocator.Persistent);

        for (int i = 0; i < num_particle; ++i)
        {
            Particle p = new Particle();
            p.pos = init_pos[i];
            p.v = 0;
            p.C = 0;
            p.mass = 1.0f;
            ps[i] = p;

            if(usePrefab)
            {
                float2 worldPos = (p.pos - grid_res/2.0f) / grid_res * render_range;
                prefab_list.Add((Instantiate(particle_prefab, new Vector3(worldPos.x, worldPos.y, -.1f), Quaternion.identity) as GameObject).GetComponent<Transform>());
            }
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
            if(useJobs)
            {
                SimulationJobs();
            }
            else
            {
                ClearGrid();
                P2G_2round();
                UpdateGrid();
                G2P();
            }
        }

        if(!usePrefab)
            sim_renderer.RenderFrame(ps);
    }

    void SimulationJobs()
    {
        Profiler.BeginSample("Clear Grid");
        new ClearGridJob()
        {
            grid = grid
        }.Schedule(num_cell, 256).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("P2G first parallel");
        new P2GJob_first_parallel()
        {
            grid = grid,
            ps = ps,
        }.Schedule(num_particle, (int)(init_edge / spacing)).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("P2G second parallel");
        new P2GJob_second_parallel()
        {
            grid = grid,
            ps = ps,
        }.Schedule(num_particle, (int)(init_edge / spacing)).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("Update Grid");
        new UpdateGridJob()
        {
            grid = grid
        }.Schedule(num_cell, 256).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("G2P");
        JobHandle G2PJob = new G2PJob()
        {
            grid = grid,
            ps = ps,
        }.Schedule(num_particle, (int)(init_edge / spacing));
        Profiler.EndSample();

        if(usePrefab)
        {
            Profiler.BeginSample("Prefab Render");
            new PrefabRenderer()
            {
                ps = ps,
                render_range = render_range
            }.Schedule(particle_array, G2PJob).Complete();
            Profiler.EndSample();
        }
        else
        {
            G2PJob.Complete();
        }
    }

    #region BurstCompileJobs
    [BurstCompile]
    struct ClearGridJob : IJobParallelFor
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
    unsafe struct P2GJob_first_parallel : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Cell> grid;
        [ReadOnly]
        public NativeArray<Particle> ps;
        public void Execute(int i)
        {
            float2* w = stackalloc float2[3];
            Particle p = ps[i];

            // The cell that particle falls in
            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    // current traversed cell
                    uint2 current_cell_idx = math.uint2(cell_idx.x + x - 1, cell_idx.y + y - 1);
                    float2 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                    float2 Q = math.mul(p.C, cell_dist);

                    float mass_contribute = weight * p.mass;

                    int index_1d = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;
                    Cell c = grid[index_1d];

                    c.mass += mass_contribute;
                    c.v += mass_contribute * (p.v + Q);

                    grid[index_1d] = c;
                }
            }
        }
    }

    [BurstCompile]
    unsafe struct P2GJob_second_parallel : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Cell> grid;
        [ReadOnly]
        public NativeArray<Particle> ps;
        public void Execute(int i)
        {
            float2* w = stackalloc float2[3];
            Particle p = ps[i];

            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float density = 0.0f;
            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    int index_1d = (int)(cell_idx.x + x - 1) * grid_res + (int)(cell_idx.y + y - 1);
                    density += grid[index_1d].mass * weight; // m_0 / h^3 in this case h = 1
                }
            }
            float volume = p.mass / density;

            float pressure = math.max(-0.1f, eos_stiffness * (math.pow(density / rest_density, eos_power) - 1));
            float2x2 stress = math.float2x2(
                -pressure, 0,
                0, -pressure
            );

            float2x2 velocity_gradient = p.C;
            float2x2 velocity_gradient_T = math.transpose(velocity_gradient);
            float2x2 strain = velocity_gradient + velocity_gradient_T;
            //float trace = strain.c1.x + strain.c0.y;
            //strain.c0.y = strain.c1.x = trace;

            float2x2 viscosity_term = dynamic_viscosity * strain;
            stress += viscosity_term;

            float2x2 eq_16_term_0 = -volume * 4 * stress * dt;

            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;

                    uint2 current_cell_idx = math.uint2(cell_idx.x + x - 1, cell_idx.y + y - 1);
                    float2 cell_dist = (current_cell_idx - p.pos) + 0.5f;

                    int index_1d = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;
                    Cell c = grid[index_1d];

                    float2 momentum = math.mul(eq_16_term_0 * weight, cell_dist);
                    c.v += momentum;

                    grid[index_1d] = c;
                }
            }
        }
    }


    [BurstCompile]
    struct UpdateGridJob : IJobParallelFor
    {
        public NativeArray<Cell> grid;

        public void Execute(int i)
        {
            Cell c = grid[i];
            if (c.mass > 0)
            {
                c.v /= c.mass;
                c.v += dt * math.float2(0, gravity);

                /*
                int x = i / grid_res;
                int y = i % grid_res;
                if (x < 2 || x > grid_res - 3)
                {
                    c.v.x = 0;
                }
                if (y < 2 || y > grid_res - 3)
                {
                    c.v.y = 0;
                }
                */

                grid[i] = c;
            }
        }
    }

    [BurstCompile]
    unsafe struct G2PJob : IJobParallelFor
    {
        [ReadOnly]public NativeArray<Cell> grid;
        public NativeArray<Particle> ps;
        public void Execute(int i)
        {
            Particle p = ps[i];

            p.v = 0;

            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;
            float2* w = stackalloc float2[3];

            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float2x2 B = 0;


            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;

                    uint2 current_cell_idx = math.uint2(cell_idx.x + x - 1, cell_idx.y + y - 1);
                    float2 cell_dist = (current_cell_idx - p.pos) + 0.5f;

                    int index_1d = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;

                    float2 weighted_velocity = grid[index_1d].v * weight;

                    float2x2 term = math.float2x2(weighted_velocity * cell_dist.x, weighted_velocity * cell_dist.y);

                    B += term;
                    p.v += weighted_velocity;
                }
            }
            p.C = B * 4;
            p.pos += p.v * dt;
            p.pos = math.clamp(p.pos, 1, grid_res - 2);

            float2 x_n = p.pos + p.v;
            float wall_min = 3;
            float wall_max = (float)grid_res - 4;
            if (x_n.x < wall_min) p.v.x += wall_min - x_n.x;
            if (x_n.x > wall_max) p.v.x += wall_max - x_n.x;
            if (x_n.y < wall_min) p.v.y += wall_min - x_n.y;
            if (x_n.y > wall_max) p.v.y += wall_max - x_n.y;
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

        public void Execute(int i, TransformAccess transform)
        {
            Particle p = ps[i];
            transform.position = math.float3((p.pos - grid_res/2)/grid_res * render_range , -.1f);
        }
    }

    #endregion

    #region singleThread
    float2[] QuadraticWeight(uint2 index, float2 diff)
    {
        float2[] weight = new float2[3];

        weight[0] = 0.5f * math.pow(0.5f - diff, 2);
        weight[1] = 0.75f - math.pow(diff, 2);
        weight[2] = 0.5f * math.pow(0.5f + diff, 2);

        return weight;
    }
    void ClearGrid()
    {
        for (int i = 0; i < num_cell; ++i)
        {
            Cell c = grid[i];

            c.v = 0;
            c.mass = 0;

            grid[i] = c;
        }
    }

    void P2G_2round()
    {
        //first round: scatter mass to grid
        for (int i = 0; i < num_particle; ++i)
        {
            Particle p = ps[i];

            // The cell that particle falls in
            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;
            float2[] w = QuadraticWeight(cell_idx, cell_diff);

            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    // current traversed cell
                    uint2 current_cell_idx = math.uint2(cell_idx.x + x - 1, cell_idx.y + y - 1);
                    float2 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                    float2 Q = math.mul(p.C, cell_dist);

                    float mass_contribute = weight * p.mass;

                    int index_1d = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;
                    Cell c = grid[index_1d];

                    c.mass += mass_contribute;
                    c.v += mass_contribute * (p.v + Q);

                    grid[index_1d] = c;
                }
            }
        }

        //second round
        for (int i = 0; i < num_particle; ++i)
        {
            Particle p = ps[i];

            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;
            float2[] w = QuadraticWeight(cell_idx, cell_diff);

            float density = 0.0f;
            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    int index_1d = (int)(cell_idx.x + x - 1) * grid_res + (int)(cell_idx.y + y - 1);
                    density += grid[index_1d].mass * weight; // m_0 / h^3 in this case h = 1
                }
            }
            float volume = p.mass / density;

            float pressure = math.max(-0.1f, eos_stiffness * (math.pow(density / rest_density, eos_power) - 1));
            float2x2 stress = math.float2x2(
                -pressure, 0,
                0, -pressure
            );

            float2x2 strain = p.C;
            float trace = strain.c1.x + strain.c0.y;
            strain.c0.y = strain.c1.x = trace;

            float2x2 viscosity_term = dynamic_viscosity * strain;
            stress += viscosity_term;

            float2x2 eq_16_term_0 = -volume * 4 * stress * dt;

            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;

                    uint2 current_cell_idx = math.uint2(cell_idx.x + x - 1, cell_idx.y + y - 1);
                    float2 cell_dist = (current_cell_idx - p.pos) + 0.5f;

                    int index_1d = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;
                    Cell c = grid[index_1d];

                    float2 momentum = math.mul(eq_16_term_0 * weight, cell_dist);
                    c.v += momentum;

                    grid[index_1d] = c;
                }
            }
        }
    }

    void UpdateGrid()
    {
        for (int i = 0; i < num_cell; ++i)
        {
            Cell c = grid[i];
            if (c.mass > 0)
            {
                c.v /= c.mass;
                c.v += dt * math.float2(0, gravity);

                int x = i / grid_res;
                int y = i % grid_res;
                if (x < 2 || x > grid_res - 3)
                {
                    c.v.x = 0;
                }
                if (y < 2 || y > grid_res - 3)
                {
                    c.v.y = 0;
                }

                grid[i] = c;
            }
        }
    }

    void G2P()
    {
        for (int i = 0; i < num_particle; ++i)
        {
            Particle p = ps[i];

            p.v = 0;

            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;
            float2[] w = QuadraticWeight(cell_idx, cell_diff);

            float2x2 B = 0;


            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;

                    uint2 current_cell_idx = math.uint2(cell_idx.x + x - 1, cell_idx.y + y - 1);
                    float2 cell_dist = (current_cell_idx - p.pos) + 0.5f;

                    int index_1d = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;

                    float2 weighted_velocity = grid[index_1d].v * weight;

                    float2x2 term = math.float2x2(weighted_velocity * cell_dist.x, weighted_velocity * cell_dist.y);

                    B += term;
                    p.v += weighted_velocity;
                }
            }
            p.C = B * 4;
            p.pos += p.v * dt;
            p.pos = math.clamp(p.pos, 1, grid_res - 2);

            float2 x_n = p.pos + p.v;
            float wall_min = 3;
            float wall_max = (float)grid_res - 4;
            if (x_n.x < wall_min) p.v.x += wall_min - x_n.x;
            if (x_n.x > wall_max) p.v.x += wall_max - x_n.x;
            if (x_n.y < wall_min) p.v.y += wall_min - x_n.y;
            if (x_n.y > wall_max) p.v.y += wall_max - x_n.y;

            ps[i] = p;
        }
    }
    #endregion

    void OnDestroy()
    {
        ps.Dispose();
        grid.Dispose();
        if(usePrefab)
            particle_array.Dispose();
    }
}
