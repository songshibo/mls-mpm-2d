using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;
using Unity.Mathematics;
using Unity.Collections;

public class Custom_MLS_MPM : MonoBehaviour
{
    struct Particle
    {
        public float2 pos;
        public float2 v;
        public float mass;
    };

    struct Cell
    {
        public float2 v;
        public float mass;
    };

    const int grid_res = 64;
    const int cell_num = grid_res * grid_res;
    int particle_num = 0;
    NativeArray<Particle> particles;
    NativeArray<Cell> grids;
    private void Start() 
    {
        List<float2> tmp = new List<float2>();
        const float spacing = 1.0f;
        float2 box = math.float2(16, 16);
        float2 sp = math.float2(grid_res / 2.0f, grid_res / 2.0f);
        // box center:(32,32),spacing: 1, generate 16 * 16 particles 
        for (float i = sp.x - box.x/2.0f; i < sp.x + box.x/2.0f; i += spacing)
        {
            for (float j = sp.y - box.y/2.0f; j < sp.y + box.y/2.0f; j+= spacing)
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
            p.v = math.float2(0.0f, Random.value);
            p.mass = 1.0f;
            particles[i] = p;
        }

        grids = new NativeArray<Cell>(cell_num, Allocator.Persistent);
        for (int i = 0; i < cell_num; ++i)
        {
            grids[i] = new Cell();
        }
    }

    private void Update() 
    {
        Simulate();
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
        for (int i = 0; i < 1; ++i)
        {
            Particle p = particles[i];

            // cell_index is the pivot of the cell which currently occupied by this particle
            uint2 cell_index = (uint2)p.pos;
            // cell_diff is a vector point from the center of the cell to the particle position
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            Debug.Log("pos:"+p.pos+"\n"+"cell index:"+cell_index+"\n"+"cell_diff"+cell_diff);
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
                    // float2 cell_dist = (current_cell_idx - p.pos) + 0.5f;

                    float mass_contribute = weight * p.mass;
                    // 1D grid index
                    int index = (int)current_cell_idx.x * grid_res + (int)current_cell_idx.y;
                    Cell cell = grids[index];
                    cell.mass += mass_contribute;
                    // !notice
                    // cell.v currently stored the momentum of particle, mass * velocity 
                    cell.v += mass_contribute * p.v;
                    grids[index] = cell;
                }
            }
        }
    }
}
