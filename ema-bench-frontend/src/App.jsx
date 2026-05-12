import './App.css'

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
  {
    title: 'Coordinated intervention',
    body: 'Teams must allocate roles, avoid redundant work, and complete urgent objectives across space and time.',
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
          <a href="#platform">Platform</a>
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
          <p className="eyebrow">Embodied Multi-Agent Decision-Making</p>
          <h1>EMA-Bench</h1>
          <p className="hero-copy">
            A high-fidelity simulation benchmark for evaluating how embodied
            agent teams perceive, reason, and coordinate in dynamic disaster
            response environments.
          </p>
          <div className="hero-actions" aria-label="Primary links">
            <a href="#video" className="primary-link">
              Watch Demonstration
            </a>
            <a href="#evaluation" className="secondary-link">
              View Evaluation
            </a>
          </div>
        </div>
      </section>

      <section className="overview band">
        <div className="section-heading">
          <p className="eyebrow">Research Problem</p>
          <h2>Dynamic disasters demand proactive multi-agent reasoning.</h2>
        </div>
        <p className="lead">
          Disaster response settings such as fire rescue are path-dependent,
          time-critical, and partially observable. A delayed decision can block
          paths, destroy targets, or remove the feasibility of later rescue
          operations. EMA-Bench shifts evaluation from static task completion to
          anticipatory, team-level decision quality under evolving risk.
        </p>
      </section>

      <section className="video-section band" id="video">
        <div className="section-heading">
          <p className="eyebrow">Demonstration</p>
          <h2>Self-progressing fire response in simulation.</h2>
        </div>
        <div className="video-frame">
          <video
            src={demonstrationVideo}
            controls
            playsInline
            preload="metadata"
            aria-label="EMA-Bench demonstration video"
          />
        </div>
      </section>

      <section className="platform band" id="platform">
        <div className="section-heading">
          <p className="eyebrow">Platform Design</p>
          <h2>Agents act while the environment keeps changing.</h2>
        </div>
        <div className="principle-grid">
          {principles.map((principle) => (
            <article className="principle-card" key={principle.title}>
              <h3>{principle.title}</h3>
              <p>{principle.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="evaluation band" id="evaluation">
        <div className="section-heading">
          <p className="eyebrow">Evaluation Framework</p>
          <h2>Three complementary dimensions measure process quality.</h2>
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