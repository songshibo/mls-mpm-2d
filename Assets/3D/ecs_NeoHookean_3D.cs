using Unity.Mathematics;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Burst;
using System.Runtime.InteropServices;
using UnityEngine.Profiling;

public class ecs_NeoHookean_3D : MonoBehaviour
{
    struct Particle
    {
        public float3 pos;
        public float3 v;
        public float3x3 C; //affine momentum matrix
        public float3x3 F;
        public float3x3 Fe;
        public float3x3 Fp;
        public float3x3 stress;
        public Dictionary<int, float> ws;
        public float mass;
        public float volume0;
    };

    struct Cell
    {
        public float3 v;
        public float3 force;
        public float mass;
    };

    //* Simulation parameters
    const float dt = 0.2f;
    const int iterations = (int)(1 / dt);
    const float gravity = -0.25f;
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
    float init_density = 2.0f;
    float init_volume = 0.5f; 
    //* sim parametera
    public float elastic_mu = 20.0f;
    public float elastic_lambda = 10.0f;
    //* Render Setting
    [SerializeField]
    GameObject particle_prefab = null;
    List<Transform> prefab_list = null;
    float render_range = 6.0f;

    void Start()
    {
        prefab_list = new List<Transform>();
        Initialization();
    }

    void Initialization()
    {
        List<float3> init_pos = new List<float3>();
        float3 center = math.float3(grid_res / 2.0f, grid_res / 2.0f, grid_res / 2.0f);
        for (float i = center.x - num_edge / 2; i < center.x + num_edge / 2; i += spacing)
        {
            for (float j = center.y - num_edge / 2; j < center.y + num_edge / 2; j += spacing)
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
            p.mass = init_volume * init_density;
            float3x3 I = math.float3x3(
                1,0,0,
                0,1,0,
                0,0,1
            );
            p.F = I;
            p.Fe = I;
            p.Fp = I;
            p.stress = I;
            p.ws = new Dictionary<int, float>();
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
            c.force = 0;
            grid[i] = c;
        }
    }
    void Update()
    {
        
    }

    int cellIndex1D(int3 idx)
    {
        return idx.x * grid_res * grid_res + idx.y * grid_res + idx.z; 
    }
    float ComputeWeightComponent(float x)
    {
        float absX = math.abs(x);
        float w = absX < 0.5f? 0.75f - math.pow(absX, 2): -1;
        w = (w < 0)&&(absX < 1.5f)? 0.5f * math.pow(1.5f - absX, 2) : 0;
        return w;
    }
    float3 ComputeWeight(int3 xi, float3 xp)
    {
        float3 diff = xp - xi - 0.5f;
        float3 weight = 0;
        weight.x = ComputeWeightComponent(diff.x);
        weight.y = ComputeWeightComponent(diff.y);
        weight.z = ComputeWeightComponent(diff.z);
        return weight;
    }

    void ClearGrid()
    {
        for(int i = 0; i < num_cell; ++i)
        {
            Cell c = grid[i];
            c.v = 0;
            c.force = 0;
            c.mass = 0;
            grid[i] = c;
        }
    }

    void P2G()
    {
        for(int i = 0; i < num_particle; ++i)
        {
            Particle p = ps[i];
            p.ws.Clear();

            int3 idx = (int3)p.pos;
            for(int x = -1; x < 2; ++x)
            {
                for(int y = -1; y < 2; ++y)
                {
                    for(int z = -1; z < 2; ++z)
                    {
                        int3 neighbor_idx = math.int3(idx.x + x, idx.y + y, idx.z + z);
                        float3 weights = ComputeWeight(idx, p.pos);
                        float w = weights.x * weights.y * weights.z;
                        int idx_1d = cellIndex1D(neighbor_idx);

                        Cell c = grid[idx_1d];
                        c.mass += p.mass * w;
                        grid[idx_1d] = c;
                        p.ws.Add(idx_1d, w);
                    }
                }
            }
            ps[i] = p;
        }
    }
}
