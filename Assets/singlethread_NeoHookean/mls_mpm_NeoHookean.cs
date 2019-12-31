using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using System.Runtime.InteropServices;
using UnityEngine.UI;

public class mls_mpm_NeoHookean : MonoBehaviour
{
    public enum volume_type
    {
        both,
        init_volume,
        sim_volume
    }
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

    const float dt = 0.1f;
    const int iterations = (int)(1.0f / dt);
    const float gravity = -0.3f;
    const int grid_res = 64;
    const int cell_num = grid_res * grid_res;
    int particle_num = 0;
    // sim parameter
    public float elastic_mu = 20.0f;
    public float elastic_lambda = 10.0f;
    public float spacing = 0.5f;
    public int side_num = 8;
    float2 rect_center = math.float2(grid_res / 2, grid_res / 2);


    // Debug Rendering Part
    [Space]
    [ReadOnlyWhenPlaying]
    public bool debug_render = true;
    public volume_type volume_debug_type = volume_type.both;
    public GameObject p_prefab;
    public float render_range = 6.0f;
    [ReadOnlyWhenPlaying]
    public int coloum = 0;
    [ReadOnlyWhenPlaying]
    public int row = 0;
    int track_index = 0;
    List<GameObject> debug_list;
    Text text;


    NativeArray<Particle> particles;
    NativeArray<Cell> grids;
    NativeArray<float2x2> Fs;
    SimRenderer sim;

    string Convert2String(float2 input, string acc)
    {
        return input.x.ToString(acc) + "," + input.y.ToString(acc);
    }

    string Convert2String(float2x2 input, string acc)
    {
        return "[" + Convert2String(input.c0, acc) + " \n " + Convert2String(input.c1, acc) + "]";
    } 

    void Start()
    {
        Initialize();
        if(!debug_render)
        {
            sim = GameObject.FindObjectOfType<SimRenderer>();
            sim.Initialise(particle_num, Marshal.SizeOf(new Particle()));
        }
        else
        {
            text = GameObject.Find("Text").GetComponent<Text>();
            debug_list = new List<GameObject>();
            GameObject root = new GameObject("DebugRender");

            track_index = coloum * (int)math.sqrt(particle_num) + row;
            if (!(track_index < particle_num && track_index >= 0))
            {
                Debug.LogError("track_index is out of range");
            }

            for(int i = 0; i < particle_num; ++i)
            {
                float2 pos = (particles[i].pos - rect_center) / grid_res * render_range;
                debug_list.Add(Instantiate(p_prefab, new Vector3(pos.x, pos.y, 0.0f), Quaternion.identity, root.transform) as GameObject);
                if (i == track_index)
                {
                    debug_list[i].GetComponent<MeshRenderer>().material.SetColor("_Color", Color.red);
                }
            }
        }
    }
    
    void Update()
    {
        for (int i = 0; i < iterations; ++i)
        {
            Simulate();
        }

        if (!debug_render)
        {
            sim.RenderFrame(particles);
        }
        else
        {
            for(int i = 0; i < particle_num; ++i)
            {
                float2 pos = (particles[i].pos - rect_center) / grid_res * render_range;
                float2 v = particles[i].v;
                debug_list[i].transform.position = new Vector3(pos.x, pos.y, 0.0f);
                Debug.DrawLine(debug_list[i].transform.position, debug_list[i].transform.position + new Vector3(v.x, v.y, 0.0f), Color.red);
                switch(volume_debug_type)
                {
                    case volume_type.both:
                        debug_list[i].transform.localScale = Vector3.one * 10.0f * math.determinant(Fs[i]) * particles[i].volume0 * 0.6f;
                        break;
                    case volume_type.init_volume:
                        debug_list[i].transform.localScale = Vector3.one * 10.0f * particles[i].volume0 * 0.6f;
                        break;
                    case volume_type.sim_volume:
                        debug_list[i].transform.localScale = Vector3.one * 10.0f * math.determinant(Fs[i]) * 0.3f;
                        break;
                }
            }

            text.text = "Index: " + track_index + "\n"
                        + "Initial volume: " + particles[track_index].volume0.ToString("f3") + "\n"
                        + "pos: (" + Convert2String(particles[track_index].pos, "f2") + ")\n"
                        + "cell index: " + (uint2)particles[track_index].pos + "\n"
                        + "velocity: (" + Convert2String(particles[track_index].v, "f2") + ")\n"
                        + "affine momentum matrix: \n" + Convert2String(particles[track_index].C, "f2")+ "\n"
                        + "deformation gradient: \n" + Convert2String(Fs[track_index], "f2") + "\n" + "Determinant:" + math.determinant(Fs[track_index]).ToString("f3");

            
            /* mouse selection (hard to use)
            //if (Input.GetMouseButtonDown(0))
            //{
            //     Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            //     RaycastHit hit;

            //     if (Physics.Raycast(ray, out hit))
            //     {
            //         GameObject objectHit = hit.transform.gameObject;
            //         int index = debug_list.IndexOf(objectHit);
            //         Debug.Log(
            //             "Hit Particle Index: " + index + "\n"
            //             + "pos: " + particles[index].pos + "\n"
            //             + "cell index: " + (uint2)particles[index].pos + "\n"
            //             + "velocity: " + particles[index].v + "\n"
            //             + "affine momentum matrix: " + particles[index].C + "\n"
            //             + "deformation gradient: " + Fs[index]
            //         );
            //     }
            // }
            */
        }
    }

    private void OnDestroy() 
    {
        particles.Dispose();
        grids.Dispose();
        Fs.Dispose();
    }

    private void Initialize()
    {
        List<float2> tmp = new List<float2>();

        for(float i = rect_center.x - side_num/2; i < rect_center.x + side_num/2; i += spacing)
        {
            for(float j = rect_center.y - side_num/2; j < rect_center.y + side_num/2; j += spacing)
            {
                tmp.Add(math.float2(i, j));
            }
        }

        // particle initialization
        particle_num = tmp.Count;
        particles = new NativeArray<Particle>(particle_num, Allocator.Persistent);
        Fs = new NativeArray<float2x2>(particle_num, Allocator.Persistent);
        for (int i = 0; i < particle_num; ++i)
        {
            Particle p = new Particle();
            p.pos = tmp[i];
            p.v = 0;
            p.C = 0;
            p.mass = 1.0f;
            particles[i] = p;

            Fs[i] = math.float2x2(
                1, 0,
                0, 1
            );
        }

        //grid initialization
        grids = new NativeArray<Cell>(cell_num, Allocator.Persistent);
        for(int i = 0; i < cell_num; ++i)
        {
            Cell c = new Cell();
            c.v = 0;
            grids[i] = c;
        }

        //scatter mass to grid
        P2G();

        //per-particle volume estimate
        for(int i = 0; i < particle_num; ++i)
        {
            Particle p = particles[i];

            uint2 cell_index = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            float2[] w = new float2[3];
            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            float density = 0.0f;

            for(int x = 0; x < 3; ++x)
            {
                for(int y = 0; y < 3; ++y)
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

    private void Simulate()
    {
        //clear grid
        ClearGrid();
        //P2G
        P2G();
        //grid velocity update
        UpdateGrid();
        //G2P
        G2P();
    }

    private void ClearGrid()
    {
        for(int i = 0; i < cell_num; ++i)
        {
            Cell c = grids[i];

            c.mass = 0;
            c.v = 0;

            grids[i] = c;
        }
    }

    private void P2G()
    {
        for(int i = 0; i < particle_num; ++i)
        {
            Particle p = particles[i];

            float2x2 F = Fs[i];

            float2x2 stress = 0;

            float J = math.determinant(F);
            float volume = p.volume0 * J;
            
            float2x2 F_T = math.transpose(F);
            float2x2 inv_F_T = math.inverse(F_T);
            float2x2 F_minus_inv_F_T = F - inv_F_T;

            //equation-48
            float2x2 P = elastic_mu * F_minus_inv_F_T + elastic_lambda * math.log(J) * inv_F_T;
            //equation-38
            stress = (1.0f / J) * math.mul(P, F_T);

            // Mp = (1/4) * (delta_x)^2, delta_x = 1 --> Mp^-1 = 4
            float2x2 eq_16_term_0 = -volume * 4 * stress * dt;

            uint2 cell_index = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            float2[] w = new float2[3];
            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            for(uint x = 0; x < 3; ++x)
            {
                for(uint y = 0; y < 3; ++y)
                {
                    float weight = w[x].x * w[y].y;
                    uint2 current_index = math.uint2(cell_index.x + x - 1, cell_index.y + y - 1);
                    float2 dist = (current_index - p.pos) + 0.5f;
                    float2 Q = math.mul(p.C, dist);

                    int index_1d = (int)current_index.x * grid_res + (int)current_index.y;
                    Cell c  = grids[index_1d];

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

    private void UpdateGrid()
    {
        for(int i = 0; i < cell_num; ++i)
        {
            Cell c = grids[i];

            if(c.mass > 0)
            {
                c.v /= c.mass;
                c.v += dt * math.float2(0, gravity);

                //boundary condition
                int x = i / grid_res;
                int y = i % grid_res;
                if (x < 2 || x > grid_res - 3) {c.v.x = 0;}
                if (y < 2 || y > grid_res - 3) {c.v.y = 0;}
            }

            grids[i] = c;
        }
    }

    private void G2P()
    {
        for(int i = 0; i < particle_num; ++i)
        {
            Particle p = particles[i];
            // reset particle velocity
            p.v = 0;

            uint2 cell_index = (uint2)p.pos;
            float2 cell_diff = (p.pos - cell_index) - 0.5f;
            float2[] w = new float2[3];
            w[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            w[1] = 0.75f - math.pow(cell_diff, 2);
            w[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            // C = B * (D^-1), D=1/4 * (delta_x)^2 * I when using quadratic interpolation
            float2x2 B = 0;
            for (uint x = 0; x < 3; ++x) {
                for (uint y = 0; y < 3; ++y) {
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
}
