using Unity.Mathematics;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using System.Runtime.InteropServices;

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

    const int grid_res = 64;
    const int num_cell = grid_res * grid_res;

    const float dt = 0.2f;
    const int iterations = (int)(1/dt);
    const float gravity = -0.2f;
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

    SimRenderer sim_renderer;

    void Start()
    {
        Initialize();
        sim_renderer = GameObject.FindObjectOfType<SimRenderer>();
        sim_renderer.Initialise(num_particle, Marshal.SizeOf(new Particle())); 
    }

    void Initialize()
    {
        List<float2> init_pos = new List<float2>();
        float2 center = math.float2(grid_res/2, grid_res/2);
        for(float i = center.x - init_edge/2; i < center.x + init_edge/2; i += spacing)
        {
            for(float j = center.y - init_edge/2; j < center.y + init_edge/2; j += spacing)
            {
                init_pos.Add(math.float2(i, j));
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
            p.mass = 1.0f;
            ps[i] = p;
        }

        grid = new NativeArray<Cell>(num_cell, Allocator.Persistent);
        for(int i = 0; i < num_cell; ++i)
        {
            Cell c = new Cell();
            c.v = 0;
            grid[i] = c;
        }
    }

    void Update()
    {
        for(int i = 0; i < iterations; ++i)
        {
            ClearGrid();
            P2G_2round();
            UpdateGrid();
            G2P();
        }

        sim_renderer.RenderFrame(ps);
    }

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
        for(int i = 0; i < num_cell; ++i)
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
        for(int i = 0; i < num_particle; ++i)
        {
            Particle p = ps[i];

            // The cell that particle falls in
            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;
            float2[] w = QuadraticWeight(cell_idx, cell_diff);

            for(uint x = 0; x < 3; ++x)
            {
                for(uint y = 0; y < 3; ++y)
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
        for(int i = 0; i < num_particle; ++i)
        {
            Particle p = ps[i];

            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;
            float2[] w = QuadraticWeight(cell_idx, cell_diff);

            float density = 0.0f;
            for(uint x = 0; x < 3; ++x)
            {
                for(uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    int index_1d = (int)(cell_idx.x + x - 1) * grid_res + (int)(cell_idx.y + y - 1);
                    density += grid[index_1d].mass * weight; // m_0 / h^3 in this case h = 1
                }
            }
            float volume = p.mass / density;

            float pressure  = math.max(-0.1f, eos_stiffness * (math.pow(density/rest_density, eos_power) - 1));
            float2x2 stress = math.float2x2(
                -pressure, 0,
                0, -pressure
            );

            float2x2 strain = p.C;
            float trace = strain.c1.x + strain.c0.y;
            strain.c0.y = strain.c1.x = trace;

            float2x2 viscosity_term = dynamic_viscosity * strain;
            stress += viscosity_term;

            float2x2 eq_16_term_0 = - volume * 4 * stress * dt;

            for(uint x = 0; x < 3; ++x)
            {
                for(uint y = 0; y < 3; ++y)
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
        for(int i = 0; i < num_cell; ++i)
        {
            Cell c = grid[i];
            if(c.mass > 0)
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
        for(int i = 0; i < num_particle; ++i)
        {
            Particle p = ps[i];

            p.v = 0;

            uint2 cell_idx = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_idx) - 0.5f;
            float2[] w = QuadraticWeight(cell_idx, cell_diff);

            float2x2 B = 0;
            

            for(uint x = 0; x < 3; ++x)
            {
                for(uint y = 0; y < 3; ++y)
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
            if(x_n.x < wall_min) p.v.x += wall_min - x_n.x;
            if(x_n.x > wall_max) p.v.x += wall_max - x_n.x;
            if(x_n.y < wall_min) p.v.y += wall_min - x_n.y;
            if(x_n.y > wall_max) p.v.y += wall_max - x_n.y;

            ps[i] = p;
        }
    }

    void OnDestroy()
    {
        ps.Dispose();
        grid.Dispose();
    }
}
