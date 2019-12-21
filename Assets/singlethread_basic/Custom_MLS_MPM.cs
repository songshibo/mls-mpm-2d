using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;
using Unity.Mathematics;
using Unity.Collections;
using System.Runtime.InteropServices;

public class Custom_MLS_MPM : MonoBehaviour
{

    // simulation part
    struct Particle
    {
        public float2 pos;
        public float2 v;
        public float2x2 C;// affine momentum matrix
        public float mass;
    };

    struct Cell
    {
        public float2 v;
        public float mass;
    };
    const float dt = 1.0f;
    const int iterations = (int)(1.0f / dt);
    const float gravity = -0.05f;
    const int grid_res = 64;
    const int cell_num = grid_res * grid_res;
    int particle_num = 0;
    NativeArray<Particle> particles;
    NativeArray<Cell> grids;
    SimRenderer sim_renderer;
    private void Start() 
    {
        List<float2> tmp = new List<float2>();
        const float spacing = 1.0f;
        const int edge = 32;
        float2 boxCenter = math.float2(grid_res / 2.0f, grid_res / 2.0f);
        // box center:(32,32),spacing: 1, generate 16 * 16 particles 
        for (float i = boxCenter.x - edge/2; i < boxCenter.x + edge/2; i += spacing)
        {
            for (float j = boxCenter.y - edge/2; j < boxCenter.y + edge/2; j += spacing)
            {
                tmp.Add(math.float2(i, j));
            }
        }

        // get particle number
        particle_num = tmp.Count;

        particles = new NativeArray<Particle>(particle_num, Allocator.Persistent);
        for (int i = 0; i < particle_num; ++i)
        {
            Particle p = new Particle();
            p.pos = tmp[i];
            p.v = math.float2(Random.value * 2 - 1, Random.value) ;
            p.mass = 1.0f;
            particles[i] = p;
        }

        grids = new NativeArray<Cell>(cell_num, Allocator.Persistent);
        for (int i = 0; i < cell_num; ++i)
        {
            grids[i] = new Cell();
        }

        sim_renderer = GameObject.FindObjectOfType<SimRenderer>();
        sim_renderer.Initialise(particle_num, Marshal.SizeOf(new Particle()));
    }

    private void Update() 
    {
        for (int i = 0; i < iterations; ++i)
        {
            Simulate();
        }
        sim_renderer.RenderFrame(particles);
    }

    private void OnDestroy() 
    {
        particles.Dispose();
        grids.Dispose();
    }

    void Simulate()
    {
        // reset grid
        for (int i = 0; i < cell_num; ++i)
        {
            // can not directly modify NativeArray
            Cell c = grids[i];
            c.mass = 0;
            c.v = 0;
            grids[i] = c;
        }

        // P2G
        // the pivot of cell is left-bottom corner(can be seen as (0,0))
        // and particles are initialized on that corner
        // but the center in a cell should be (0.5, 0.5)
        for (int i = 0; i < particle_num; ++i)
        {
            Particle p = particles[i];

            // cell_index is the pivot of the cell which currently occupied by this particle
            uint2 cell_index = (uint2)p.pos;
            // cell_diff is a vector point from the center of the cell to the particle position
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            // quadratic interpolation
            // need 3 weights
            float2[] w = new float2[3];
            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            for (uint x = 0; x < 3; ++x)
            {
                for (uint y =0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    // pivot of current cell
                    uint2 current_cell_idx = math.uint2(cell_index.x + x - 1, cell_index.y + y - 1);
                    // the distance between current cell center and particles
                    // a vector point from particle to the center of cell
                    float2 cell_dist = (current_cell_idx - p.pos) + 0.5f;
                    float2 Q = math.mul(p.C, cell_dist);

                    float mass_contribute = weight * p.mass;
                    // 1D grid index
                    int index = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;
                    Cell cell = grids[index];
                    cell.mass += mass_contribute;
                    // !notice
                    // cell.v currently stored the momentum of particle, mass * velocity 
                    cell.v += mass_contribute * (p.v + Q);
                    grids[index] = cell;
                }
            }
        }

        // grid velocity
        for (int i = 0; i < cell_num; ++i)
        {
            Cell cell = grids[i];

            if (cell.mass > 0)
            {
                cell.v /= cell.mass;
                cell.v += dt * math.float2(0, gravity);

                //handle boundary
                int x = i / grid_res;
                int y = i % grid_res;
                if (x < 2 || x > grid_res - 3) 
                {
                    cell.v.x = 0;
                }
                if (y < 2 || y > grid_res -3)
                {
                    cell.v.y = 0;
                }
            }

            grids[i] = cell;
        }

        //G2P
        for (int i = 0; i < particle_num; ++i)
        {
            Particle p = particles[i];

            p.v = 0;

            uint2 cell_index = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            float2[] w = new float2[3];
            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float2x2 B = 0;
            for (uint x = 0; x < 3; ++x) {
                for (uint y = 0; y < 3; ++y) {
                    float weight = w[x].x * w[y].y;

                    uint2 current_cell_idx = math.uint2(cell_index.x + x - 1, cell_index.y + y - 1);
                    int index = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;
                    
                    float2 dist = (current_cell_idx - p.pos) + 0.5f;
                    float2 weighted_velocity = grids[index].v * weight;

                    // APIC paper equation 10, constructing inner term for B
                    float2x2 term = math.float2x2(weighted_velocity * dist.x, weighted_velocity * dist.y);

                    B += term;

                    p.v += weighted_velocity;
                }
            }
            p.C = B * 4;

            // advect particles
            p.pos += p.v * dt;
            p.pos = math.clamp(p.pos, 1, grid_res - 2);

            particles[i] = p;
        }
    }
}
