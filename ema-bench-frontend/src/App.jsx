import './App.css'
import paperPdfUrl from './EMA_Bench.pdf?url'
import overviewImageUrl from './picture/overview.jpg?url'
import agentframeworkImageUrl from './picture/framework.jpg?url'
import environmentassetsImageUrl from './picture/assets_rooms.jpg?url'
import npcassetsImageUrl from './picture/assets_npc.png?url'

const demonstrationVideo =
  'https://ema-benchmark.oss-cn-beijing.aliyuncs.com/demonstration.webm'

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

function App() {
  return (
    <main>
      <header className="site-header" aria-label="EMA-Bench navigation">
        <a className="brand" href="#top" aria-label="EMA-Bench home">
          EMA-Bench
        </a>
        <nav>
          <a href="#challenge">Challenges</a>
          <a href="#evaluation">Evaluation</a>
          <a href="#findings">Findings</a>
          <a href="#contributions">Contributions</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <video
          className="hero-video"
          src={demonstrationVideo}
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
              Dataset&Simulator
            </a>
          </div>
        </div>
      </section>

{/* Abstract*/}
      <section className="abstract band">
        <div className="section-heading">
          {/* <p className="eyebrow">Research Problem</p> */}
          <h2>Abstract</h2>
        </div>
        <p className="lead">
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

{/* Challenges*/}
      <section className="challenge band" id="challenge">
        <div className="section-heading">
          <h2>Challenges</h2>
          {/* <p className="eyebrow">What makes dynamic environments challenging for multi-agent systems?</p> */}
        </div>
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
        </section>

{/* Agent-Environment Interaction*/}
      <section className="agent-environment-interaction band" id="agent-environment-interaction">
        <div className="section-heading">
          <h2>Agent-Environment Interaction</h2>
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
           
           <div className="text-block">
             <h3>Observation & Action Space</h3>
             <p>
               Agents perceive via <b>Visual Interface</b> (RGB-D frames) or <b>Symbolic Interface</b> (Object-centric metadata). [cite: 247, 248]
               The action space includes 17 primitives across Navigation, Hazard Intervention, and Active Perception. [cite: 253, 268]
             </p>
           </div>
        </div>
      </section>

{/* ENVIRONMENT & ASSETS*/}
      <section className="environment-assets band" id="environment-assets">
        <div className="section-heading">
          <h2>Environment & Assets</h2>
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
      </section>

      <section className="evaluation band" id="evaluation">
        <div className="section-heading">
          <h2>System Overview</h2>
          <p className="eyebrow">How does EMA-Bench evaluate multi-agent systems?</p>
        </div>
        <div className="track-list">
          {evaluationTracks.map((track) => (
            <article className="track" key={track.title}>
              <span>{track.number}</span>
              <div>
                <h3>{track.title}</h3>
                <p>{track.body}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="findings band" id="findings">
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
      </section>
    </main>
  )
}

export default App