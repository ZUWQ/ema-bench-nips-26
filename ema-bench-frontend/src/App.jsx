import './App.css'
// import paperPdfUrl from './EMA_Bench.pdf?url'
import overviewImageUrl from './picture/overview.jpg?url'
import agentframeworkImageUrl from './picture/framework.jpg?url'
import environmentassetsImageUrl from './picture/assets_rooms.jpg?url'
import npcassetsImageUrl from './picture/assets_npc.png?url'
// Hugging Face: https://huggingface.co/datasets/<user>/<repo>/resolve/<revision>/<path>
const HF_DEMO_VIDEO_BASE =
  'https://huggingface.co/datasets/ZWQ1103/temp_video/resolve/main'

const paperPdfUrl = `${HF_DEMO_VIDEO_BASE}/EMA_Bench.pdf`
const demonstrationVideoUrl = `${HF_DEMO_VIDEO_BASE}/demo.webm`
const demo1ImageUrl = `${HF_DEMO_VIDEO_BASE}/01DHP.webm`
const demo2ImageUrl = `${HF_DEMO_VIDEO_BASE}/02RCE.webm`
const demo3ImageUrl = `${HF_DEMO_VIDEO_BASE}/03IPD.webm`

// Local copies (use with Vite `import ... from './picture/xxx.webm?url'` if you bundle demos in-repo):
// const demonstrationVideoUrl = new URL('./picture/demo.webm', import.meta.url).href
// const demo1ImageUrl = new URL('./picture/01DHP.webm', import.meta.url).href
// const demo2ImageUrl = new URL('./picture/02RCE.webm', import.meta.url).href
// const demo3ImageUrl = new URL('./picture/03IPD.webm', import.meta.url).href

const principles = [
  {
    title: 'Self-progressing fire dynamics',
    body: 'Hazards spread and intensify over time, forcing agents to reason before the environment outruns the response.',
  },
  {
    title: 'Action-environment coupling',
    body: 'Movement, exploration, and suppression change both immediate progress and future scene states.',
  },
  {
    title: 'Partial observability',
    body: 'Agents operate from local views and must build shared situational awareness through exploration.',
  },
]
// 3 Rigorous Properties (Sec 1 / Sec 3.1.2)
const challenges = [
  {
    number: '01',
    title: 'Dynamic Hazard Processes',
    body: 'The environment possesses an intrinsic engine that drives monotonic degradation independent of agent actions. [cite: 56]',
  },
  {
    number: '02',
    title: 'Rapidly Cumulative Escalation',
    body: 'A "snowball effect" where hazard severity compounds over time; early delays result in disproportionately higher costs. [cite: 57, 60]',
  },
  {
    number: '03',
    title: 'Interactive Path-Dependency',
    body: "Future trajectories are not pre-determined but jointly reshaped by autonomous progression and agent interventions. [cite: 82]",
  },
]

const evaluationTracks = [
  {
    number: '01',
    title: 'Foundational Task Execution',
    body: 'Measures navigation, target localization, fire suppression, and rescue-related actions inside the simulator.',
  },
  {
    number: '02',
    title: 'Environmental Exploration and Understanding',
    body: 'Evaluates how agents discover hazards, critical paths, fire sources, and evolving scene states under incomplete information.',
  },
  {
    number: '03',
    title: 'Collaborative Efficiency',
    body: 'Quantifies whether agent teams divide labor, share useful information, and finish high-priority objectives before failures become irreversible.',
  },
]

const findings = [
  'Current agents often miss high-priority risks when fire spreads rapidly.',
  'Models struggle to balance immediate rewards against long-term environmental consequences.',
  'Multi-agent teams frequently show redundant exploration and delayed coordination.',
  'Short-horizon perception is stronger than sustained team-level strategy.',
]

const comparisonData = [
  {
    simulator: "AI2-THOR",
    autonomousDynamics: false,
    irreversible: false,
    partialObs: true,
    heteroMAS: false,
    dynamicIntervention: false,
    llmVlmReady: "-",
    fidelity: "Ultra",
  },
  {
    simulator: "Habitat 3.0",
    autonomousDynamics: true,
    irreversible: false,
    partialObs: true,
    heteroMAS: true,
    dynamicIntervention: false,
    llmVlmReady: "-",
    fidelity: "High",
  },
  {
    simulator: "Habitat-MAS",
    autonomousDynamics: false,
    irreversible: false,
    partialObs: false,
    heteroMAS: true,
    dynamicIntervention: false,
    llmVlmReady: "LLM",
    fidelity: "High",
  },
  {
    simulator: "GRUtopia",
    autonomousDynamics: true,
    irreversible: false,
    partialObs: true,
    heteroMAS: true,
    dynamicIntervention: false,
    llmVlmReady: "LLM & VLM",
    fidelity: "High",
  },
  {
    simulator: "RoCoBench",
    autonomousDynamics: false,
    irreversible: false,
    partialObs: false,
    heteroMAS: true,
    dynamicIntervention: false,
    llmVlmReady: "LLM",
    fidelity: "Mid",
  },
  {
    simulator: "RFUniverse",
    autonomousDynamics: false,
    irreversible: true,
    partialObs: false,
    heteroMAS: false,
    dynamicIntervention: false,
    llmVlmReady: "-",
    fidelity: "Ultra",
  },
  {
    simulator: "VIKI-R",
    autonomousDynamics: false,
    irreversible: false,
    partialObs: false,
    heteroMAS: true,
    dynamicIntervention: false,
    llmVlmReady: "LLM & VLM",
    fidelity: "High",
  },
  {
    simulator: "Virtual Home",
    autonomousDynamics: false,
    irreversible: false,
    partialObs: true,
    heteroMAS: false,
    dynamicIntervention: false,
    llmVlmReady: "-",
    fidelity: "Ultra",
  },
  {
    simulator: "RoBoGen",
    autonomousDynamics: false,
    irreversible: false,
    partialObs: false,
    heteroMAS: false,
    dynamicIntervention: false,
    llmVlmReady: "LLM",
    fidelity: "Ultra",
  },
  {
    simulator: "Virtual Community",
    autonomousDynamics: true,
    irreversible: false,
    partialObs: false,
    heteroMAS: true,
    dynamicIntervention: false,
    llmVlmReady: "LLM",
    fidelity: "Ultra",
  },
  {
    simulator: "FD-Bench",
    autonomousDynamics: true,
    irreversible: true,
    partialObs: true,
    heteroMAS: false,
    dynamicIntervention: false,
    llmVlmReady: "LLM",
    fidelity: "Symbolic",
  },
  {
    simulator: "Hazard Challenge",
    autonomousDynamics: true,
    irreversible: true,
    partialObs: true,
    heteroMAS: false,
    dynamicIntervention: false,
    llmVlmReady: "LLM",
    fidelity: "High",
  },
  {
    simulator: "Crew-Wildfire",
    autonomousDynamics: true,
    irreversible: true,
    partialObs: true,
    heteroMAS: true,
    dynamicIntervention: true,
    llmVlmReady: "LLM",
    fidelity: "Voxel",
  },
  {
    simulator: "EMA-Bench",
    autonomousDynamics: true,
    irreversible: true,
    partialObs: true,
    heteroMAS: true,
    dynamicIntervention: true,
    llmVlmReady: "LLM & VLM",
    fidelity: "Ultra",
    ours: true,
  },
];


function App() {
  return (
    <main>
      <header className="site-header" aria-label="EMA-Bench navigation">
        <a className="brand" href="#top" aria-label="EMA-Bench home">
          EMA-Bench
        </a>
        <nav>
          <a href="#overview">Overview</a>
          <a href="#challenge">Challenges</a>
          <a href="#agent-environment-interaction">Agent-Environment Interaction</a>
          <a href="#environment-assets">Environment & Assets</a>
          <a href="#demonstration">Demonstration</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <video
          className="hero-video"
          src={demonstrationVideoUrl}
          autoPlay
          muted
          loop
          playsInline
          aria-hidden="true"
        />
        <div className="hero-shade" />
        <div className="hero-content">
        <p className="eyebrow">NeurIPS 2026 Anonymous Submission</p>
        <h1>EMA-Bench</h1>
          <p className="hero-copy">
            {/* Decision-making in dynamic environments with propagating and compounding hazards, where environmental changes occur independently of agent intervention. */}
            A Benchmark for Embodied Multi-Agent Decision-Making in Dynamic Environments
          </p>
          <div className="hero-actions" aria-label="Primary links">
            <a
              href={paperPdfUrl}
              className="primary-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Paper&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            </a>
            <a href="https://anonymous.4open.science/r/ema-bench-nips-26-4E30" className="secondary-link">
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            </a>
            <a href="https://huggingface.co/datasets/EMAS4Rescue/ema-bench-26" className="primary-link">
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dataset&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            </a>
            <a href="https://drive.google.com/file/d/1wgos_ZZG2T6XC8TU00DXmWDLy1ua4tHv/view?usp=drive_link" className="secondary-link">
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Simulator&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            </a>
          </div>
        </div>
      </section>

{/* Abstract*/}
      <section className="abstract band" lang="en">
        <div className="section-heading">
          {/* <p className="eyebrow">Research Problem</p> */}
          <h2>Abstract</h2>
        </div>
        <p className="lead abstract-lead">
        Embodied multi-agent systems are vital for high-risk disaster response, yet they struggle in dynamic environments characterized by rapidly hazard escalation and path-dependent dynamics. The rapid compounding of hazards in these settings demands a shift from reactive execution to proactive reasoning to effectively anticipate environmental dynamics. Furthermore, the extreme time sensitivity of these scenarios makes multi-agent cooperation a functional necessity, as agents must coordinate their efforts to prevent the disaster from outpacing the team's capacity. To address this, we introduce EMA-Bench, a high-fidelity simulation platform designed to evaluate multi-agent coordination within self-progressing fire. EMA-Bench facilitates interactions where agent actions directly influence the environmental progression under strict temporal urgency and partial observability. We propose a structured evaluation framework spanning foundational task execution, environmental exploration, and collaborative efficiency. Our empirical analysis of state-of-the-art multimodal foundation model-based agents highlights a significant deficiency in their ability to handle time-sensitive trade-offs and irreversible state transitions. These findings reveal a substantial gap in current embodied intelligence and establish a rigorous foundation for future research in resilient multi-agent coordination.
        </p>
      </section>

{/* Overview*/}
      <section className="overview band" id="overview">
        <div className="section-heading">
          <h2>Overview</h2>
        </div>
        <div className="overview-image-wrap">
          <img
            src={overviewImageUrl}
            alt="EMA-Bench overview figure"
            className="overview-image"
            loading="lazy"
            decoding="async"
          />
        </div>
        <div className="figure-placeholder">Figure 1: Multi-agent collaborative firefighting and rescue framework. We employ a heterogeneous team of firefighting and rescue robots to coordinate fire suppression and human evacuation. The panels illustrate the core components of the system: collaborative task execution in an indoor fire scenario (center), fire growth dynamics (left), the rescue process (bottom), and a communication example demonstrating inter-agent coordination (right).</div>
        </section>

{/* Challenges + demonstrations (demos on the right) */}
      <section className="challenge band" id="challenge">
        <div className="section-heading">
          <h2>Challenges</h2>
          <p className="eyebrow">Demonstrations</p>
          {/* <p className="eyebrow">What makes dynamic environments challenging for multi-agent systems?</p> */}
        </div>
        <div className="challenge-layout">
          <div className="track-list">
            {challenges.map((challenge) => (
              <article className="track" key={challenge.title}>
                <span>{challenge.number}</span>
                <div>
                  <h3>{challenge.title}</h3>
                  <p>{challenge.body}</p>
                </div>
              </article>
            ))}
          </div>
          <div className="challenge-demos" id="demonstration">
            <div className="challenge-demo-block" id="demonstration-fire">
              {/* <h4>Demonstration_fire</h4> */}
              <div className="video-frame video-frame--compact">
                <video
                  src={demo1ImageUrl}
                  controls
                  playsInline
                  preload="metadata"
                  aria-label="EMA-Bench demonstration_fire video"
                />
              </div>
            </div>
            <div className="challenge-demo-block" id="demonstration-extinguish">
              {/* <h4>Demonstration_extinguish</h4> */}
              <div className="video-frame video-frame--compact">
                <video
                  src={demo3ImageUrl}
                  controls
                  playsInline
                  preload="metadata"
                  aria-label="EMA-Bench demonstration_extinguish video"
                />
              </div>
            </div>
            <div className="challenge-demo-block" id="demonstration-rescue">
              {/* <h4>Demonstration_rescue</h4> */}
              <div className="video-frame video-frame--compact">
                <video
                  src={demo2ImageUrl}
                  controls
                  playsInline
                  preload="metadata"
                  aria-label="EMA-Bench demonstration_rescue video"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

{/* Agent-Environment Interaction*/}
      <section className="agent-environment-interaction band" id="agent-environment-interaction">
        <div className="section-heading">
          <h2>Agent&nbsp;-&nbsp;Environment&nbsp;Interaction</h2>
        </div>
        <div className="overview-image-wrap">
          <img
            src={agentframeworkImageUrl}
            alt="EMA-Bench agent-environment interaction figure"
            className="overview-image"
            loading="lazy"
            decoding="async"
          />
        </div>
        <div className="content-split">
           {/* [IMAGE: Insert Figure 2 - Overview of the agent-environment interaction [cite: 298]] */}
           <div className="figure-placeholder">Figure 2: Overview of the agent–environment interaction. Each agent operates in a closed loop of perception, decision-making, action execution, and memory update, while sharing information with other agents.</div>
           
           {/* <div className="text-block">
             <h3>Observation & Action Space</h3>
             <p>
               Agents perceive via <b>Visual Interface</b> (RGB-D frames) or <b>Symbolic Interface</b> (Object-centric metadata). [cite: 247, 248]
               The action space includes 17 primitives across Navigation, Hazard Intervention, and Active Perception. [cite: 253, 268]
             </p>
           </div> */}
        </div>
      </section>

{/* ENVIRONMENT & ASSETS*/}
      <section className="findings band" id="environment-assets">
        <div className="section-heading">
          <h2>Environment&nbsp;&&nbsp;Assets</h2>
        </div>
        <div className="overview-image-wrap">
          <img
            src={environmentassetsImageUrl}
            alt="EMA-Bench overview figure"
            className="overview-image"
            loading="lazy"
            decoding="async"
          />
        </div>
        <div className="figure-placeholder">Figure 4: Overview of EMA-Bench Environments. A 4 × 7 grid showcasing 28 base room variations across 7 functional categories. Each scene integrates high-fidelity assets, including 2 types of quadrupedal robots, 5 civilian models, and diverse combustible furniture. The combinatorial nature of these layouts, combined with randomized hazard and victim placement, ensures a vast and unpredictable task space.</div>
        <div className="overview-image-wrap">
          <img
            src={npcassetsImageUrl}
            alt="EMA-Bench NPC assets figure"
            className="overview-image"
            loading="lazy"
            decoding="async"
          />
        </div>
        <div className="figure-placeholder">Figure 5: Overview of Agent and NPC Assets. EMA-Bench features a heterogeneous set of interactive entities: (left) Five distinct civilian NPC models representing trapped victims, each integrated with dynamic health monitoring and surface soot accumulation effects to provide realistic visual-physical feedback during missions.(right) The rescue fleet consisting of the firefighting robot, specialized for active hazard suppression, and the rescue robot, optimized for rapid navigation and victim extraction.</div>

      </section>

      <section className="demonstration band" id="demonstration">
        <div className="section-heading">
          <h2>Demonstration</h2>
        </div>
        <div className="video-frame">
          <video
            src={demonstrationVideoUrl}
            controls
            playsInline
            preload="metadata"
            aria-label="EMA-Bench demonstration video"
          />
        </div>
      </section>

{/* Comparison Table — same chrome as Related work table */}
      <section
        className="related-work band"
        id="simulator-comparison"
        aria-labelledby="simulator-comparison-heading"
      >
        <div className="section-heading">
          <h2 id="simulator-comparison-heading">Simulator comparison</h2>
          <p className="eyebrow">Capabilities vs. related simulation platforms</p>
        </div>
        <div className="related-work-table-wrap">
          <table className="related-work-table related-work-table--comparison">
            <caption className="related-work-caption">
              Checkmarks indicate supported capability; em dash denotes not supported or not applicable.
              Highlighted row: EMA-Bench (this work).
            </caption>
            <thead>
              <tr>
                <th scope="col">
                  <span className="related-work-th-l1">Simulator</span>
                  <br />
                  <span className="related-work-th-l2">&nbsp;</span>
                </th>
                <th scope="col">
                  <span className="related-work-th-l1">Autonomous</span>
                  <br />
                  <span className="related-work-th-l2">Dynamics</span>
                </th>
                <th scope="col">
                  <span className="related-work-th-l1">Irreversible</span>
                  <br />
                  <span className="related-work-th-l2">&nbsp;</span>
                </th>
                <th scope="col">
                  <span className="related-work-th-l1">Partial</span>
                  <br />
                  <span className="related-work-th-l2">Obs.</span>
                </th>
                <th scope="col">
                  <span className="related-work-th-l1">Hetero-</span>
                  <br />
                  <span className="related-work-th-l2">MAS</span>
                </th>
                <th scope="col">
                  <span className="related-work-th-l1">Dynamic</span>
                  <br />
                  <span className="related-work-th-l2">Intervention</span>
                </th>
                <th scope="col">
                  <span className="related-work-th-l1">LLM &amp; VLM</span>
                  <br />
                  <span className="related-work-th-l2">Ready</span>
                </th>
                <th scope="col">
                  <span className="related-work-th-l1">Fidelity</span>
                  <br />
                  <span className="related-work-th-l2">&nbsp;</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {comparisonData.map((item, idx) => (
                <tr
                  key={`${item.simulator}-${idx}`}
                  className={item.ours ? 'related-work-table-row--ours' : undefined}
                >
                  <td>{item.simulator}</td>
                  <td>{item.autonomousDynamics ? '✅' : '—'}</td>
                  <td>{item.irreversible ? '✅' : '—'}</td>
                  <td>{item.partialObs ? '✅' : '—'}</td>
                  <td>{item.heteroMAS ? '✅' : '—'}</td>
                  <td>{item.dynamicIntervention ? '✅' : '—'}</td>
                  <td>{item.llmVlmReady}</td>
                  <td>{item.fidelity}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>


      {/* <section className="findings band" id="findings">
        <div className="section-heading">
          <p className="eyebrow">Empirical Findings</p>
          <h2>State-of-the-art agents remain brittle under urgency.</h2>
        </div>
        <ul className="finding-list">
          {findings.map((finding) => (
            <li key={finding}>{finding}</li>
          ))}
        </ul>
      </section>

      <section className="contributions band" id="contributions">
        <div className="section-heading">
          <p className="eyebrow">Contributions</p>
          <h2>A reproducible benchmark for resilient embodied intelligence.</h2>
        </div>
        <p className="lead">
          EMA-Bench introduces a structured testbed for self-progressing fire
          dynamics, time-sensitive objectives, irreversible state transitions,
          and collaborative resource allocation. The benchmark exposes where
          current multimodal foundation model-based agents fall short and
          creates a foundation for future research on reliable disaster
          response coordination.
        </p>
      </section> */}
    </main>
  )
}

export default App