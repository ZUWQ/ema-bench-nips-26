import './App.css'

// Placeholder for the Main Demo Video (Sec 1)
const mainDemoVideo = 'https://link-to-your-anonymous-video/main_demo.mp4'

// 3 Rigorous Properties (Sec 1 / Sec 3.1.2)
const properties = [
  {
    title: 'Dynamic Hazard Processes',
    body: 'The environment possesses an intrinsic engine that drives monotonic degradation independent of agent actions. [cite: 56]',
  },
  {
    title: 'Rapidly Cumulative Escalation',
    body: 'A "snowball effect" where hazard severity compounds over time; early delays result in disproportionately higher costs. [cite: 57, 60]',
  },
  {
    title: 'Interactive Path-Dependency',
    body: "Future trajectories are not pre-determined but jointly reshaped by autonomous progression and agent interventions. [cite: 82]",
  },
]

function App() {
  return (
    <main>
      <header className="site-header" aria-label="EMA-Bench navigation">
        <a className="brand" href="#top">EMA-Bench</a>
        <nav>
          <a href="#challenges">Challenges</a>
          <a href="#framework">Framework</a>
          <a href="#environment">Environment</a>
          <a href="#results">Results</a>
        </nav>
      </header>

      {/* HERO SECTION (Revised for Anonymity and Impact) */}
      <section className="hero" id="top">
        <video className="hero-video" src={mainDemoVideo} autoPlay muted loop playsInline />
        <div className="hero-shade" />
        <div className="hero-content">
          <p className="eyebrow">NeurIPS 2026 Anonymous Submission</p>
          <h1>EMA-Bench: A Benchmark for Embodied Multi-Agent Decision-Making in Dynamic Environments</h1>
          <p className="hero-copy">
            A high-fidelity simulation platform designed to evaluate multi-agent coordination within 
            self-progressing fire. [cite: 45]
          </p>
          
          {/* Quick Links */}
          <div className="hero-actions">
            <a href="#" className="btn-link">Paper (PDF)</a>
            <a href="#" className="btn-link">Code (Anonymous)</a>
            <a href="#" className="btn-link">Data (Hugging Face)</a>
          </div>
        </div>
      </section>

      {/* 1. THREE CORE CHALLENGES (Sec 1) */}
      <section className="band" id="challenges">
        <div className="section-heading">
          <h2>The Rigorous Properties of EMA-Bench</h2>
          <p>Shifting from reactive execution to proactive reasoning to anticipate environmental dynamics. [cite: 43]</p>
        </div>
        <div className="principle-grid">
          {properties.map((p) => (
            <article className="principle-card" key={p.title}>
              <h3>{p.title}</h3>
              <p>{p.body}</p>
              {/* [PLACEHOLDER: Insert small loop video showing the specific property] */}
            </article>
          ))}
        </div>
      </section>

      {/* 2. SYSTEM FRAMEWORK (Sec 3.4 / Sec 4) */}
      <section className="band" id="framework">
        <div className="section-heading">
          <h2>Agent-Environment Interaction</h2>
        </div>
        <div className="content-split">
           {/* [IMAGE: Insert Figure 2 - Overview of the agent-environment interaction [cite: 298]] */}
           <div className="figure-placeholder">Figure 2: Overview of the agent-environment interaction</div>
           
           <div className="text-block">
             <h3>Observation & Action Space</h3>
             <p>
               Agents perceive via <b>Visual Interface</b> (RGB-D frames) or <b>Symbolic Interface</b> (Object-centric metadata). [cite: 247, 248]
               The action space includes 17 primitives across Navigation, Hazard Intervention, and Active Perception. [cite: 253, 268]
             </p>
             {/* [IMAGE: Insert Observation/Action Space Figure/Table [cite: 115]] */}
           </div>
        </div>
      </section>

      {/* 3. ENVIRONMENT & ASSETS (Appendix A) */}
      <section className="band light" id="environment">
        <div className="section-heading">
          <h2>High-Fidelity Assets & Stochastic Generation</h2>
        </div>
        <div className="asset-grid">
          <div className="asset-item">
             {/* [IMAGE: Insert Figure 4 - Overview of EMA-Bench Environments [cite: 675]] */}
             <p><b>Combinatorial Scene Diversity:</b> 28 base room variations yielding 16,384 unique permutations. [cite: 675, 709]</p>
          </div>
          <div className="asset-item">
             {/* [IMAGE: Insert Figure 5 - Overview of Agent and NPC Assets [cite: 691]] */}
             <p><b>Active Entities:</b> Heterogeneous quadrupedal robots and 5 civilian models with dynamic health monitoring. [cite: 691]</p>
          </div>
        </div>
      </section>

      {/* 4. MECHANISM SPOTLIGHTS (Sub-videos as requested) */}
      <section className="band" id="mechanisms">
        <div className="section-heading">
          <h2>Mechanism Spotlights</h2>
        </div>
        <div className="video-grid">
           <article>
             {/* [VIDEO: Fire Spread/Natural Burn Mechanism [cite: 195]] */}
             <p>Fire Growth & Spatial Propagation</p>
           </article>
           <article>
             {/* [VIDEO: Victim Health Degradation with soot effects [cite: 886]] */}
             <p>Civilian Health Dynamics</p>
           </article>
           <article>
             {/* [VIDEO: Intervention / Extinguishing Fire [cite: 261]] */}
             <p>Dynamic Multi-agent Intervention</p>
           </article>
        </div>
      </section>

      {/* 5. EXPERIMENTAL RESULTS (Sec 5) */}
      <section className="band" id="results">
        <div className="section-heading">
          <h2>Main Results: The Grounding Crisis</h2>
          <p>Current models fall short of human expert performance in handling irreversible dynamics. [cite: 112, 435]</p>
        </div>
        <div className="results-container">
           {/* [IMAGE: Insert Main Results Table 2 [cite: 417]] */}
           <div className="chart-placeholder">Table 2: Comparison of Symbolic vs. Vision-Based Agents</div>
           
           <div className="content-split">
             {/* [IMAGE: Insert Figure 3 - Damage progression curves [cite: 458]] */}
             <div className="chart-placeholder">Figure 3: Response Lag Comparison</div>
             
             <div className="text-block">
               <h3>Failure Patterns</h3>
               <ul>
                 <li><b>Perceptual Disorientation:</b> Vision models fail to identify objects from pixels. [cite: 460]</li>
                 <li><b>Coordination Deadlock:</b> Physical obstruction at bottlenecks like doorways. [cite: 461, 469]</li>
                 <li><b>Reasoning-Action Mismatch:</b> Valid strategies but invalid output coordinates. [cite: 470]</li>
               </ul>
               {/* [VIDEO: The "Consensus Fracture" Failure Case [cite: 1327]] */}
             </div>
           </div>
        </div>
      </section>

      <footer className="site-footer">
        <div className="band">
          <p>© 2026 EMA-Bench Team. NeurIPS 2026 Anonymous Submission.</p>
          <pre className="bibtex-block">
            {`@inproceedings{zu2026emabench,
  title={EMA-Bench: A Benchmark for Embodied Multi-Agent Decision-Making in Dynamic Environments},
  author={Anonymous Authors},
  booktitle={40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year={2026}
}`}
          </pre>
        </div>
      </footer>
    </main>
  )
}

export default App
