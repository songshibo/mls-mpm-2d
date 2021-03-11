using Unity.Mathematics;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Burst;
using System.Runtime.InteropServices;
using UnityEngine.Profiling;

public class NeoHookean_ECS : MonoBehaviour
{
    struct Particle
    {
        public float2 pos;
        public float2 v;
        public float2x2 C; //affine momentum matrix
        public float mass;
        public float volume0;
    };

    struct Cell
    {
        public float2 v;
        public float mass;
    };

    const float dt = 0.05f;
    const int iterations = (int)(1.0f / dt);
    const float gravity = -0.3f;
    const int grid_res = 128;
    const int cell_num = grid_res * grid_res;
    int num_particle = 0;

    // sim parameter
    [SerializeField]
    float elastic_mu = 20.0f;
    [SerializeField]
    float elastic_lambda = 10.0f;
    [SerializeField]
    [ReadOnlyWhenPlaying]
    float spacing = 0.5f;
    [SerializeField]
    [ReadOnlyWhenPlaying]
    int side_num = 8;

    NativeArray<Particle> particles;
    NativeArray<float2x2> Fs;
    NativeArray<Cell> grids;
    SimRenderer sim;

    void Start()
    {
        Initialize();
        sim = GameObject.FindObjectOfType<SimRenderer>();
        sim.Initialise(num_particle, Marshal.SizeOf(new Particle()));
    }

    void Update()
    {
        for (int i = 0; i < iterations; ++i)
        {
            Simulate();
        }

        sim.RenderFrame(particles);
    }

    void OnDestroy()
    {
        particles.Dispose();
        grids.Dispose();
        Fs.Dispose();
    }

    void Initialize()
    {
        List<float2> tmp = new List<float2>();
        float2 rect_center = math.float2(grid_res / 2, grid_res / 2);
        for (float i = rect_center.x - side_num / 2; i < rect_center.x + side_num / 2; i += spacing)
        {
            for (float j = rect_center.y - side_num / 2; j < rect_center.y + side_num / 2; j += spacing)
            {
                tmp.Add(math.float2(i, j));
            }
        }

        num_particle = tmp.Count;
        particles = new NativeArray<Particle>(num_particle, Allocator.Persistent);
        Fs = new NativeArray<float2x2>(num_particle, Allocator.Persistent);
        for (int i = 0; i < num_particle; ++i)
        {
            Particle p = new Particle();
            p.pos = tmp[i];
            p.v = 0;
            p.C = 0;
            p.mass = 1.0f;
            Fs[i] = math.float2x2(
                1, 0,
                0, 1
            );
            particles[i] = p;
        }

        grids = new NativeArray<Cell>(cell_num, Allocator.Persistent);
        for (int i = 0; i < cell_num; ++i)
        {
            Cell c = new Cell();
            c.v = 0;
            grids[i] = c;
        }

        new P2G()
        {
            grids = grids,
            particles = particles,
            Fs = Fs,
            elastic_mu = elastic_mu,
            elastic_lambda = elastic_lambda
        }.Schedule(num_particle, side_num * 2).Complete();

        new P2G_initial_volume()
        {
            particles = particles,
            grids = grids
        }.Schedule(num_particle, side_num * 2).Complete();
    }

    void Simulate()
    {
        new ClearGrid()
        {
            grids = grids
        }.Schedule(cell_num, grid_res).Complete();

        new P2G()
        {
            grids = grids,
            particles = particles,
            Fs = Fs,
            elastic_mu = elastic_mu,
            elastic_lambda = elastic_lambda
        }.Schedule(num_particle, side_num * 2).Complete();

        new UpdateGrid()
        {
            grids = grids,
            gravity = gravity
        }.Schedule(cell_num, grid_res).Complete();

        new G2P()
        {
            particles = particles,
            Fs = Fs,
            grids = grids
        }.Schedule(num_particle, side_num * 2).Complete();
    }

    #region simulation job
    [BurstCompile]
    unsafe struct P2G : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Cell> grids;
        [ReadOnly]
        public NativeArray<Particle> particles;
        [ReadOnly]
        public NativeArray<float2x2> Fs;
        [ReadOnly]
        public float elastic_mu;
        [ReadOnly]
        public float elastic_lambda;
        public void Execute(int i)
        {
            Particle p = particles[i];
            float2x2 F = Fs[i];
            float2x2 stress = 0;

            float J = math.determinant(F);
            float volume = p.volume0 * J;

            float2x2 F_T = math.transpose(F);
            float2x2 inv_F_T = math.inverse(F_T);
            float2x2 F_minus_inv_F_T = F - inv_F_T;

            float2x2 P = elastic_mu * F_minus_inv_F_T + elastic_lambda * math.log(J) * inv_F_T;
            stress = (1.0f / J) * math.mul(P, F_T);

            float2x2 eq_16_term_0 = -volume * 4 * stress * dt;

            uint2 cell_index = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            float2* w = stackalloc float2[3];
            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    uint2 current_index = math.uint2(cell_index.x + x - 1, cell_index.y + y - 1);
                    float2 dist = (current_index - p.pos) + 0.5f;
                    float2 Q = math.mul(p.C, dist);

                    int index_1d = (int)current_index.x * grid_res + (int)current_index.y;
                    Cell c = grids[index_1d];

                    float mass_contribute = weight * p.mass;
                    c.mass += mass_contribute;

                    c.v += mass_contribute * (p.v + Q);

                    float2 momentum = math.mul(eq_16_term_0 * weight, dist);
                    c.v += momentum;
                    // current cell.v is w_ij * (dt * M^-1 * p.volume * p.stress + p.mass * p.C)

                    grids[index_1d] = c;
                }
            }
        }
    }

    [BurstCompile]
    unsafe struct P2G_initial_volume : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Particle> particles;
        [ReadOnly]
        public NativeArray<Cell> grids;
        public void Execute(int i)
        {
            Particle p = particles[i];

            uint2 cell_index = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            float2* w = stackalloc float2[3];
            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float density = 0.0f;

            for (int x = 0; x < 3; ++x)
            {
                for (int y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    int index_1d = ((int)cell_index.x + (x - 1)) * grid_res + ((int)cell_index.y + (y - 1));
                    density += grids[index_1d].mass * weight;
                }
            }

            float volume = p.mass / density;
            p.volume0 = volume;

            particles[i] = p;
        }
    }

    [BurstCompile]
    struct ClearGrid : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Cell> grids;
        public void Execute(int i)
        {
            Cell c = grids[i];

            c.mass = 0;
            c.v = 0;

            grids[i] = c;
        }
    }

    [BurstCompile]
    struct UpdateGrid : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Cell> grids;
        [ReadOnly]
        public float gravity;
        public void Execute(int i)
        {
            Cell c = grids[i];

            if (c.mass > 0)
            {
                c.v /= c.mass;
                c.v += dt * math.float2(0, gravity);

                //boundary condition
                int x = i / grid_res;
                int y = i % grid_res;
                if (x < 2 || x > grid_res - 3) { c.v.x = 0; }
                if (y < 2 || y > grid_res - 3) { c.v.y = 0; }
            }

            grids[i] = c;
        }
    }

    [BurstCompile]
    unsafe struct G2P : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<Particle> particles;
        [NativeDisableParallelForRestriction]
        public NativeArray<float2x2> Fs;
        [ReadOnly]
        public NativeArray<Cell> grids;
        public void Execute(int i)
        {
            Particle p = particles[i];
            // reset particle velocity
            p.v = 0;

            uint2 cell_index = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            float2* w = stackalloc float2[3];
            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            // C = B * (D^-1), D=1/4 * (delta_x)^2 * I when using quadratic interpolation
            float2x2 B = 0;
            for (uint x = 0; x < 3; ++x)
            {
                for (uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;

                    uint2 current_index = math.uint2(cell_index.x + x - 1, cell_index.y + y - 1);
                    int index = (int)current_index.x * grid_res + (int)current_index.y;

                    float2 dist = (current_index - p.pos) + 0.5f;
                    float2 weighted_velocity = grids[index].v * weight;

                    // APIC paper equation 10, constructing inner term for B
                    float2x2 term = math.float2x2(weighted_velocity * dist.x, weighted_velocity * dist.y);

                    B += term;

                    p.v += weighted_velocity;
                }
            }
            p.C = B * 4;
            p.pos += p.v * dt;
            p.pos = math.clamp(p.pos, 1, grid_res - 2);

            //deformation gradient update eqation-181
            float2x2 F_new = math.float2x2(
                1, 0,
                0, 1
            );
            F_new += dt * p.C;
            Fs[i] = math.mul(F_new, Fs[i]);

            particles[i] = p;
        }
    }
    #endregion
}
