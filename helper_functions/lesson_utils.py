import logging
import os
import random
import re
from typing import Any, Dict, Tuple

from helper_functions.llm_client import get_client

logger = logging.getLogger(__name__)

# List of topics
topics = [  
    "How can I be more productive?", "How to improve my communication skills?", "How to be a better leader?",  
    "How are electric vehicles less harmful to the environment?", "How can I think clearly in adverse scenarios?",  
    "What are the tenets of effective office politics?", "How to be more creative?", "How to improve my problem-solving skills?",  
    "How to be more confident?", "How to be more empathetic?", "What can I learn from Boyd, the fighter pilot who changed the art of war?",  
    "How can I seek the mentorship I want from key influential people", "How can I communicate more effectively?",  
    "Give me suggestions to reduce using filler words when communicating highly technical topics?",  
    "How to apply the best game theory concepts in getting ahead in office poilitics?", "What are some best ways to play office politics?",  
    "How to be more persuasive, assertive, influential, impactful, engaging, inspiring, motivating, captivating and convincing in my communication?",  
    "What are the top 8 ways the tit-for-tat strategy prevails in the repeated prisoner's dilemma, and how can these be applied to succeed in life and office politics?",  
    "What are Chris Voss's key strategies from *Never Split the Difference* for hostage negotiations, and how can they apply to workplace conflicts?",  
    "How can tactical empathy (e.g., labeling emotions, mirroring) improve outcomes in high-stakes negotiations?",  
    "What is the 'Accusations Audit' technique, and how does it disarm resistance in adversarial conversations?",  
    "How do calibrated questions (e.g., *How am I supposed to do that?*) shift power dynamics in negotiations?",  
    "When should you use the 'Late-Night FM DJ Voice' to de-escalate tension during disagreements?",  
    "How can anchoring bias be leveraged to set favorable terms in salary or deal negotiations?",  
    "What are 'Black Swan' tactics for uncovering hidden information in negotiations?",  
    "How can active listening techniques improve conflict resolution in team settings?",  
    "What non-verbal cues (e.g., tone, body language) most impact persuasive communication?",  
    "How can I adapt my communication style to different personality types (e.g., assertive vs. analytical)?",  
    "What storytelling frameworks make complex ideas more compelling during presentations?",  
    "How do you balance assertiveness and empathy when delivering critical feedback?",  
    "What are strategies for managing difficult conversations (e.g., layoffs, project failures) with grace?",  
    "How can Nash Equilibrium concepts guide decision-making in workplace collaborations?",  
    "What real-world scenarios mimic the 'Chicken Game,' and how should you strategize in them?",  
    "How do Schelling Points (focal points) help teams reach consensus without direct communication?",  
    "When is tit-for-tat with forgiveness more effective than strict reciprocity in office politics?",  
    "How does backward induction in game theory apply to long-term career or project planning?",  
    "What are examples of zero-sum vs. positive-sum games in corporate negotiations?",  
    "How can Bayesian reasoning improve decision-making under uncertainty (e.g., mergers, market entry)?",  
    "How can Boyd's OODA Loop (Observe, Orient, Decide, Act) improve decision-making under pressure?",  
    "What game theory principles optimize resource allocation in cross-functional teams?",  
    "How can the 'MAD' (Mutually Assured Destruction) concept deter adversarial behavior in workplaces?", 
    "How does Conway's Law ('organizations design systems that mirror their communication structures') impact the efficiency of IP or product design?",  
    "What strategies can mitigate the negative effects of Conway's Law on modularity in IP design (e.g., reusable components)?",  
    "How can robust handoff methodologies (e.g., structured documentation, checklists, simulations) ensure consistent IP reuse across teams?",  
    "What lessons from Toyota Production System (e.g., Kaizen, Jidoka) apply to improving silicon tape-out processes?",  
    "How can error-proofing techniques (Poka-Yoke) reduce silicon respins and design bugs?",  
    "How can you design modular silicon IP that supports plug-and-play integration across SoC projects?",  
    "What is Design for Manufacturability (DFM) in silicon, and how can it reduce yield issues?",  
    "How does Agile hardware development differ from Agile software practices?",  
    "What steps ensure high-quality documentation for reusable IP (e.g., interface specs, timing diagrams)?",  
    "How can version control and tagging strategies prevent IP compatibility issues?",  
    "What metrics best capture IP quality and readiness for reuse?",  
    "How can silicon design teams balance feature velocity with rigorous verification?",  
    "What is the role of formal verification in preventing silicon bugs in complex IPs?",  
    "How can CI/CD be adapted for hardware (e.g., automated regressions, lint, CDC checks)?",  
    "What are best practices for clock domain crossing (CDC) in mixed-signal SoCs?",  
    "How can you architect low-power designs for battery-sensitive devices (DVFS, power gating)?",  
    "What are common pitfalls in analog/digital co-design, and how can they be avoided?",  
    "How can hardware emulation platforms speed up firmware development before silicon availability?",  
    "What is the trade-off between IP customization and standardization for reuse?",  
    "How can you manage third-party IP risks (security, licensing, quality)?",  
    "What are strategies to optimize area, power, and performance in SoC design?",  
    "How can you apply root cause analysis (RCA) to post-silicon debug failures?",  
    "What is silicon bring-up, and how do you design test plans to de-risk it?",  
    "How do you set up design for testability (DFT) to improve yield and debug?",  
    "What metrics should an analog circuit design engineer track to measure career success?",  
    "How can mixed-signal simulation flows reduce integration surprises?",  
    "How do you balance analog performance targets (noise, linearity) with digital requirements (timing, area)?",  
    "What communication strategies help analog engineers influence SoC architecture decisions?",  
    "How can you explain complex analog concepts to non-experts (e.g., PMs, software engineers)?",  
    "What leadership lessons from engineering history inspire modern analog designers?",  
    "How can you mentor junior analog engineers effectively?",  
    "What negotiation tactics help secure resources for analog IP quality improvements?",  
    "How can analog design insights drive system-level innovation (e.g., sensor fusion, edge AI)?",  
    "What career strategies help analog engineers transition into technical leadership roles?",  
    "How can analog engineers showcase their impact to executives?",  
    "What are best practices for cross-team design reviews to catch issues early?",  
    "How can you design for reliability and long-term stability in analog circuits?",  
    "What lessons from Boyd's OODA Loop apply to high-stakes tape-out decisions?",  
    "How can scenario planning improve risk management in silicon programs?",  
    "What patterns make technical presentations persuasive to non-technical stakeholders?",  
    "How do high-performing teams handle late-stage design changes without chaos?",  
    "What tactics improve focus and deep work for engineers in noisy environments?",  
    "How can you create feedback loops to continuously improve analog design processes?",  
    "What strategies help analog engineers stay relevant as AI accelerators evolve?",  
    "How can analog knowledge contribute to AI hardware optimizations?",  
    "What are practical steps to reduce meeting overload while staying aligned?",  
    "How can engineers negotiate scope and timelines without harming relationships?",  
    "What habits build credibility and trust with cross-functional partners?",  
    "How can you cultivate strategic thinking about technology trends in semiconductors?",  
    "What is the best play to lead the future of semi-conductors in a world of AI?", 
    "What are highly scalable ways for Dinesh to be a changemaker in the semi-conductor space now, and in the future?", 
    "What is the best way for a senior analog engineer to elevate to a chief technical officer role in his future at Willamette? Provide next steps towards that.", 
    "What is the best way for Dinesh to go independent and start his own semiconductor company?", 
    "How can history inform decision-making?",
    "What strategic principles can we learn from the Roman Empire's military campaigns?",
    "How did the Renaissance shift thinking in art, science, and humanism?",
    "What leadership lessons emerge from the lives of great explorers like Magellan or Zheng He?",
    "How has the concept of liberty evolved from the Magna Carta to modern democracies?",
    "What can leaders learn from the resilience of ancient civilizations like Egypt or Mesopotamia?",
    "How did the Industrial Revolution reshape economies and societies globally?",
    "What parallels exist between the fall of ancient empires and modern geopolitical tensions?",
    "How did the Cold War shape technological innovation and global alliances?",
    "What can we learn from the moral and political philosophies of thinkers like Kant, Locke, or Rousseau?",
    "How did ancient trade routes like the Silk Road foster globalization before the modern era?",
    "What strategies did historical figures use to overcome adversity and resistance?",
    "How can historical breakthroughs in science guide today's ethical decisions in tech?",
    "What are the pivotal lessons from major world revolutions (American, French, etc.)?",
    "How have ideas of human rights evolved through history and influenced modern policies?",
    "How did the ancient Greeks lay foundations for democracy, philosophy, and science?",
    "What can modern businesses learn from the strategies of historical empires?",
    "How did the printing press revolutionize knowledge dissemination and what parallels exist with today's internet?",
    "What lessons can leaders draw from the failures of historical leaders?",
    "How did religious movements shape historical events and power dynamics?",
    "What can we learn about resilience from societies that rebuilt after disasters?",
    "What leadership lessons emerge from historical figures like Napoleon or Alexander the Great?",
    "How did ancient engineering feats like the Pyramids or Roman aqueducts influence modern engineering?",
    "What can modern negotiators learn from historic treaties and diplomatic successes?",
    "How did colonialism alter global power dynamics and cultural exchanges?",
    "What historical patterns repeat in modern politics and economics?",
    "How did the Enlightenment change views on reason, science, and individual rights?",
    "What lessons in ethics can we draw from historical debates on slavery, suffrage, and civil rights?",
    "How did maritime exploration transform economies and cultures in the Age of Discovery?",
    "What can the rise and fall of ancient cities teach about sustainability and urban planning?",
    "How have military strategies evolved from ancient times to modern warfare?",
    "What can we learn from the resilience and adaptability of historical innovators?",
    "How did the Great Depression shape modern economic policy and financial regulation?",
    "What can the space race teach us about competition, innovation, and national pride?",
    "How have pandemics throughout history influenced public health and societal change?",
    "What can the history of education systems teach about improving learning today?",
    "What can we learn about leadership from the founding of the United States?",
    "How did ancient law codes (like Hammurabi) influence modern legal systems?",
    "What lessons about tolerance and coexistence emerge from multiethnic empires?",
    "How did technological innovations like gunpowder or the steam engine change warfare and industry?",
    "What are the strategic lessons from historical espionage and intelligence operations?",
    "How did the abolitionist movement succeed in changing hearts, minds, and laws?",
    "What can the fall of the Berlin Wall teach about resilience and the power of people?",
    "How did the Scientific Revolution change humanity's relationship with nature?",
    "What lessons from the Great Recession apply to financial risk management today?",
    "How did ancient democracies handle corruption, and what parallels exist today?",
    "What can we learn from the cultural renaissances across different civilizations?",
    "How did the invention of the internet compare to earlier communication revolutions?",
    "What can leaders learn from the mistakes of the League of Nations?",
    "How did ancient philosophers debate ethics, and how can those debates inform modern dilemmas?",
    "What can the history of medicine teach about innovation and patient care?",
    "How did imperialism shape today's geopolitical borders and tensions?",
    "What can the history of labor movements teach about modern workplace dynamics?",
    "How did ancient navigators use science to master the seas?",
    "What can we learn from the preservation and loss of ancient knowledge?",
    "How did the rise of nation-states transform Europe in the Middle Ages?",
    "What are the enduring lessons from the Roman Republic's transition to empire?",
    "How did Enlightenment thinkers influence the French and American revolutions?",
    "What can we learn from the cultural exchanges of the Islamic Golden Age?",
    "How did industrialization impact social structures and daily life?",
    "What can the history of civil rights movements teach about persistence and strategy?",
    "How did the Renaissance change approaches to art, science, and the human experience?",
    "What can we learn from historical innovators who combined disciplines like art and engineering?",
    "How did military alliances shape the outcomes of World Wars I and II?",
    "What can we learn from historical debates on free speech and censorship?",
    "How did ancient agricultural innovations support the growth of civilizations?",
    "What can business leaders learn from the successes and failures of ancient traders?",
    "What can leaders learn from wartime decision-making and crises?",
    "How did cultural revolutions shape modern music, literature, and art?",
    "What can we learn from the historical evolution of democracy and its challenges?",
    "How did transportation innovations like railways and automobiles reshape economies and cities?",
    "What lessons can investors learn from historical market bubbles and crashes?",
    "How did the history of science shape today's research ethics and practices?",
    "What can we learn from historical peace movements and diplomacy?",
    "How did the colonial independence movements succeed against empires?",
    "What lessons from ancient and modern engineers can solve today's infrastructural challenges?",
    "How do historical communication revolutions parallel today's social media landscape?",
    "How does the Roman Empire's infrastructure compare to modern megaprojects?",
    "What can be learned from the resilience of cultures that endured conquests and disasters?",
    "How have ideas about leadership evolved from monarchies to modern corporate structures?",
    "What can we learn from history about balancing security and liberty?",
    "How can ancient philosophies guide modern debates on technology ethics?",
    "What can the history of global trade teach about modern supply chain resilience?",
    "How have societies historically responded to inequality, and what worked?",
    "What lessons about innovation emerge from the history of patents and intellectual property?",
    "How did women's roles evolve across different historical periods and cultures?",
    "What can we learn from the rise and fall of historical financial centers?",
    "How did post-war reconstruction efforts succeed or fail in rebuilding nations?",
    "What can the history of public health teach about vaccination and disease prevention?",
    "How did historical education reforms shape modern schooling?",
    "How do historical debates about privacy and surveillance inform today's issues?",
    "What can the history of propaganda teach about media literacy today?",
    "How have ideas about citizenship and rights evolved over time?",
    "What can leaders learn from historical explorations and risk-taking ventures?",
    "How did the ancient city-states balance rivalry and cooperation?",
    "What can we learn from the cultural syncretism of regions like the Mediterranean?",
    "How did historical scientific discoveries challenge prevailing worldviews?",
    "What lessons about resilience and innovation emerge from the history of technology?",
    "How did the transatlantic slave trade shape global economies and societies?",
    "What can we learn from the history of philanthropy and social reform?",
    "How did technological advances in warfare change geopolitical power balances?",
    "What lessons about governance emerge from the failures of ancient regimes?",
    "How did historical migration patterns shape cultural identities and demographics?",
    "What can we learn from the history of urban planning and architecture?",
    "How did revolutionary ideas spread before modern communication tools?",
    "What can the history of religious reform teach about social change?",
    "How did global conflicts catalyze innovation in science and medicine?",
    "What lessons about economic policy emerge from historical trade wars?",
    "How did historical leaders navigate crises and maintain legitimacy?",
    "How have legal systems evolved to balance tradition and progress?",
    "What can the history of exploration teach about curiosity and discovery?",
    "How did historical thinkers grapple with the ethics of power and authority?",
    "What can we learn from the history of education in fostering critical thinking?",
    "How did historical debates on justice influence modern legal principles?",
    "What lessons from ancient democracy apply to modern civic engagement?",
    "How did historical rebellions shape the course of empires?",
    "What can the history of science communication teach about bridging experts and the public?",
    "How have concepts of freedom and equality evolved across cultures and eras?",
    "What lessons about sustainability emerge from societies that managed natural resources well?",
    "How did historical advances in navigation and cartography change worldviews?",
    "What can we learn from the history of innovation hubs and creative cities?",
    "How did historical epidemics influence art, culture, and religion?",
    "What lessons about leadership emerge from transformative historical moments?",
    "How did historical financial crises reshape economies and regulations?",
    "What can the history of volunteerism and civic engagement teach today?",
    "How did shifts in trade routes impact the rise and fall of cities and empires?",
    "What can we learn from historical debates about technology's impact on society?",
    "How did historical figures balance personal ambition with public service?",
    "What lessons from the fall of civilizations can inform modern resilience planning?",
    "How have concepts of war and peace evolved throughout history?",
    "What can we learn from the evolution of legal rights and protections?",
    "How did historical communities organize for mutual aid and support?",
    "What lessons emerge from the history of innovation during times of constraint?",
    "How did historical leaders build and maintain coalitions?",
    "What can the history of media teach about the power of storytelling?",
    "How have scientific paradigms shifted over time, and what drove those changes?",
    "What can we learn from historical efforts to standardize systems (weights, measures, time)?",
    "How did historical thinkers approach the balance between faith and reason?",
    "What lessons from historical mentorship and apprenticeship apply today?",
    "How did historical societies handle dissent and protest?",
    "What can the history of mathematics teach about problem-solving approaches?",
    "How did historical cultures view and manage mental health?",
    "What lessons about innovation emerge from cross-cultural exchanges?",
    "How did historical financial instruments develop and spread?",
    "What can we learn from the history of environmental stewardship?",
    "How did historical leaders cultivate legitimacy and trust?",
    "What lessons about collaboration emerge from large historical projects?",
    "How did historical societies adapt to climate and environmental changes?",
    "What can we learn from the history of scientific collaboration across borders?",
    "How did historical innovations in communication reshape power structures?",
    "What lessons about resilience emerge from communities rebuilding after conflict?",
    "How did historical societies balance innovation with tradition?",
    "What can we learn from the history of public discourse and debate?",
    "How did historical leaders manage information and intelligence?",
    "What lessons about leadership emerge from historical exploration expeditions?",
    "How did historical advances in transportation change economies and cultures?",
    "What can we learn from the history of technological ethics debates?",
    "How did historical societies foster innovation through education and patronage?",
    "What lessons emerge from historical examples of peaceful resistance?",
    "How did historical thinkers approach the concept of progress?",
    "What can we learn from the evolution of governance models across civilizations?",
    "How did historical leaders use symbolism and rituals to maintain power?",
    "What lessons about crisis management emerge from historical disasters?",
    "How did historical cultures measure and value time?",
    "What can we learn from historical precedents of globalization?",
    "How did historical societies document and preserve knowledge?",
    "What lessons about diversity and inclusion emerge from multiethnic societies?",
    "How did historical leaders balance local and central governance?",
    "What can we learn from the history of scientific instrumentation?",
    "How did historical debates on morality shape laws and customs?",
    "What lessons about innovation emerge from patronage systems in history?",
    "How did historical societies handle misinformation and rumor?",
    "What can we learn from the history of philanthropy in addressing social needs?",
    "How did historical leaders manage succession and continuity?",
    "What lessons about resilience emerge from cultural renaissances?",
    "How did historical societies integrate new technologies into daily life?",
    "What can we learn from historical models of community governance?",
    "How did historical thinkers approach the relationship between humans and nature?",
    "What lessons about negotiation emerge from historical treaties and alliances?",
    "How did historical societies encourage or suppress creativity?",
    "What can we learn from the history of measurement and standardization?",
    "How did historical leaders respond to technological disruption?",
    "What lessons about leadership emerge from historical reformers and revolutionaries?",
    "How did historical societies view and manage risk?",
    "What can we learn from historical traditions of mentorship and learning?",
    "How did historical thinkers balance individual rights with collective good?",
    "What lessons about communication emerge from historical rhetoric and persuasion?",
    "How did historical societies adapt to demographic shifts and migrations?",
    "What can we learn from the history of experimentation and scientific method?",
    "How did historical leaders use information networks to maintain control?",
    "What lessons about innovation emerge from historical cross-pollination of ideas?",
    "How did historical societies construct and challenge social hierarchies?",
    "What can we learn from historical narratives of progress and decline?",
    "How did historical thinkers approach the ethics of power and leadership?",
    "What lessons about strategy emerge from historical conflicts and alliances?",
    "How did historical societies foster civic engagement and responsibility?",
    "What can we learn from the history of artistic and scientific patronage?",
    "How did historical leaders manage resource scarcity and abundance?",
    "What lessons about resilience emerge from historical recoveries after crises?",
    "How did historical societies view the role of technology in shaping destiny?",
    "What can we learn from historical approaches to education and wisdom?",
    "How did historical thinkers reconcile tradition with innovation?",
    "What lessons about governance emerge from historical experiments in democracy?",
    "How did historical societies navigate cultural identity amid change?",
    "What can we learn from historical debates about justice and fairness?",
    "How did historical leaders build institutions that lasted?",
    "What lessons about innovation emerge from constraints faced by historical inventors?",
    "How did historical societies measure and value knowledge?",
    "What can we learn from historical approaches to balancing power among branches of government?",
    "How did historical thinkers approach the concept of human progress?",
    "What lessons about leadership emerge from historical mentors and teachers?",
    "How did historical societies cultivate wisdom across generations?",
    "What can we learn from historical examples of ethical leadership?",
    "How did historical leaders manage change during transformative periods?",
    "What lessons about collaboration emerge from historical alliances?",
    "How did historical societies view the responsibilities of leadership?",
    "What can we learn from historical precedents for balancing innovation with caution?",
    "How did historical thinkers address the tension between individualism and collectivism?",
    "What lessons about resilience emerge from historical community rebuilding?",
    "How did historical societies institutionalize learning and knowledge sharing?",
    "What can we learn from historical approaches to mentorship and apprenticeship?",
    "How did historical leaders use narrative to inspire and mobilize?",
    "What lessons about governance emerge from historical federations and unions?",
    "How did historical societies handle rapid technological change?",
    "What can we learn from historical debates on ethics in science and technology?",
    "How did historical thinkers conceptualize the common good?",
    "What lessons about strategy emerge from historical power shifts?",
    "How did historical societies foster innovation ecosystems?",
    "What can we learn from historical models of leadership succession?",
    "How did historical leaders cultivate adaptability and foresight?",
    "What lessons about communication emerge from historical diplomacy?",
    "How did historical societies measure success and prosperity?",
    "What can we learn from historical examples of interdisciplinary innovation?",
    "How did historical thinkers address the moral dimensions of progress?",
    "What lessons about leadership emerge from historical statesmen and reformers?",
    "How did historical societies navigate competing priorities and trade-offs?",
    "What can we learn from historical practices that sustained cultural heritage?",
    "How did historical leaders use education as a strategic tool?",
    "What lessons about innovation emerge from historical problem-solving under pressure?",
    "How did historical societies view the responsibilities of the educated class?",
    "What can we learn from historical debates on rights and responsibilities?",
    "How did historical thinkers integrate philosophy with practical governance?",
    "What lessons about resilience emerge from historical adaptation to new eras?",
    "How did historical societies design systems to balance power and accountability?",
    "What can we learn from historical approaches to cultivating virtue and character?",
    "How did historical leaders manage diverse coalitions and interests?",
    "What lessons about innovation emerge from historical leaps in understanding?",
    "How did historical societies ensure continuity amid change?",
    "What can we learn from historical reflections on time, progress, and legacy?",
    "How did historical thinkers frame the purpose of education and knowledge?",
    "What lessons about leadership emerge from historical guides and mentors?",
    "How did historical societies institutionalize wisdom traditions?",
    "What can we learn from historical explorations of justice, power, and ethics?",
    
    # The Medici - Godfathers of the Renaissance
    "What factors enabled the Medici family to wield power without formal titles?",
    "How did the Medici patronage system transform the arts and sciences in Florence?",
    "What banking innovations did the Medici create, and how did they shape modern finance?",
    
    # Leonardo da Vinci
    "How did Leonardo's cross-disciplinary curiosity fuel his innovation?",
    "What does Leonardo's notebook practice teach about creative problem-solving?",
    
    # Rome - The Twelve Caesars
    "How did the leadership styles of Julius Caesar and Augustus differ?",
    "What led to the decline of the Julio-Claudian dynasty?",
    
    # Greek Philosophy
    "How did Aristotle's Golden Mean shape Western ethics?",
    "What are the key differences between Stoicism and Epicureanism in handling adversity?",
    
    # Alexander the Great
    "How did Alexander integrate diverse cultures within his empire?",
    "What logistical innovations enabled Alexander's rapid military campaigns?",
    
    # The Innovators by Walter Isaacson
    "What collaboration patterns in 'The Innovators' accelerated breakthroughs in computing?",
    "How did early computer pioneers balance vision with execution?",
    
    # Roman Empire
    "How did Rome transition from Republic to Empire politically and culturally?",
    "What infrastructure strategies allowed Rome to manage vast territories?",
    
    # American History
    "How did the Founding Fathers balance federal and state power in the Constitution?",
    "What leadership lessons emerge from Abraham Lincoln's Civil War strategy?",
    
    # World War II
    "What strategic decisions turned the tide in World War II's Pacific theater?",
    "How did codebreaking at Bletchley Park influence Allied success?",
    
    # Neuroscience and Psychology
    "How does neuroplasticity evidence support lifelong learning strategies?",
    "What does cognitive bias research teach about better decision-making?",
    
    # Economics and Innovation
    "How does Schumpeter's 'creative destruction' explain tech industry shifts?",
    "What factors make certain regions enduring hubs of innovation?",
    
    # Military Strategy
    "How do Sun Tzu's principles apply to modern business competition?",
    "What modern lessons come from the OODA loop in fast-moving markets?",
    
    # Renaissance and Enlightenment
    "How did the printing press disrupt information control in Europe?",
    "What Enlightenment ideals laid groundwork for modern democracy?",
    
    # Industrial Revolution
    "How did railroads reshape economic geography during the Industrial Revolution?",
    "What public health innovations emerged from industrial urbanization?",
    
    # Silicon Valley and Tech History
    "What culture and process lessons can hardware startups learn from Intel's early days?",
    "How did open-source movements change the trajectory of software innovation?",
    
    # Indian History
    "How did Chanakya's Arthashastra guide statecraft and economics in ancient India?",
    "What factors led to the rise and fall of the Mughal Empire?",
    
    # Physics and Engineering
    "How did Maxwell's equations unify electricity and magnetism conceptually?",
    "What engineering tradeoffs shaped the design of the first microprocessors?",
    
    # Space Exploration
    "How did the Apollo program manage risk under extreme time pressure?",
    "What lessons from SpaceX's iterative testing apply to other industries?",
    
    # Management and Leadership
    "How did Toyota's production system principles spread globally?",
    "What does Drucker's management philosophy advise for knowledge workers?",
    
    # Systems Thinking and Complexity
    "How can feedback loops and leverage points improve organizational change efforts?",
    "What can chaos theory teach leaders about planning in uncertainty?",
    
    # AI and Future of Work
    "How will generative AI reshape creative professions over the next decade?",
    "What policies balance AI-driven productivity with worker well-being?",
    
    # Philosophy and Ethics
    "How does utilitarianism differ from deontology in tough decision scenarios?",
    "What does virtue ethics suggest about character-building habits for leaders?",
    
    # Notorious RBG
    "What negotiation tactics did Ruth Bader Ginsburg use to build consensus on the Supreme Court?",
    "How did RBG strategically select cases to advance gender equality law?",
    
    # Indian Independence
    "How did the Indian independence movement balance nonviolent resistance with political strategy?",
    "What leadership lessons come from Sardar Patel's role in unifying princely states?",
    
    # History of Computing
    "How did the transition from vacuum tubes to transistors unlock new computing paradigms?",
    "What design decisions in early computer architecture still influence systems today?",
    
    # Naval History
    "How did Admiral Nelson's tactics at Trafalgar depart from naval convention?",
    "What can modern leaders learn from the logistics of historic naval expeditions?",
    
    # Technology Policy
    "How have antitrust approaches to tech monopolies evolved over time?",
    "What lessons from telecom regulation apply to today's internet platforms?",
    
    # Music, Art, and Creativity
    "How did jazz improvisation influence modern creativity frameworks?",
    "What can software teams learn from agile practices in film production?",
    
    # Women's Leadership
    "How did pioneers like Ada Lovelace and Grace Hopper expand computing's possibilities?",
    "What strategies helped women leaders navigate male-dominated fields historically?",
    
    # Bio Engineering
    "How did the discovery of CRISPR reshape bioengineering possibilities?",
    "What ethical frameworks guide responsible innovation in genetics?",
    
    # Modern strategy and geopolitics
    "How does supply chain resilience influence national security strategies?",
    "What lessons from past energy transitions apply to today's shift toward renewables?",
    
    # China - 1400-1840
    "How did China's tributary system manage foreign relations during the Ming dynasty?",
    "What internal factors led to the Qing dynasty's struggles with Western powers?",
    
    # Peter the Great
    "How did Peter the Great's reforms modernize Russia's military and governance?",
    "What cultural tradeoffs did Peter the Great make in westernizing Russia?",
    
    # Buddha and Hinduism
    "How did Buddhism respond to and reshape prevailing Hindu philosophies?",
    "What leadership lessons emerge from the Buddhist concept of the Middle Way?",
    
    # Roosevelt, Diplomacy, American Greatness in Century ahead
    "How did FDR's diplomatic strategy shape the post-WWII international order?",
    "What lessons does Roosevelt's leadership offer for modern global cooperation?",
    
    # Silicon IP reuse at major companies
    "What strategies do leading chip companies use to maximize IP reuse across product lines?",
    "How can standardized verification flows improve IP reuse outcomes?",
    
    # AI hardware and economics
    "How do AI accelerator design choices impact total cost of ownership in data centers?",
    "What economic models best forecast demand for AI hardware over the next decade?",
    
    # The Silk Roads
    "How did the ancient trade networks of the Silk Roads facilitate cultural exchange and technological diffusion?",
    "What does a Silk Roads perspective teach us about geopolitical power centers throughout history?",
    
    # 1491
    "What advanced civilizations existed in pre-Columbian Americas that challenge our historical narratives?",
    "How did indigenous American societies develop sophisticated agricultural and urban systems before European contact?",
    
    # The Age of Revolution
    "How did the dual revolutions (French and Industrial) fundamentally reshape European society?",
    "What economic and social factors drove the revolutionary changes across Europe from 1789-1848?",
    
    # The Ottoman Empire
    "What administrative innovations allowed the Ottoman Empire to successfully govern a diverse, multi-ethnic state?",
    "How did the Ottoman Empire's position between East and West influence its cultural development?",
    
    # Decline and Fall of the Roman Empire
    "What internal factors contributed most significantly to the Roman Empire's decline?",
    "How did the rise of Christianity influence the political transformation of the Roman Empire?",
    
    # The Warmth of Other Suns
    "How did the Great Migration of African Americans transform both Northern and Southern American society?",
    "What personal stories from the Great Migration reveal about systemic racism and individual resilience?",
    
    # The Peloponnesian War
    "What strategic and leadership lessons can be learned from Athens' and Sparta's conflict?",
    "How did democratic Athens' political system influence its military decisions during the war?",
    
    # The Making of the Atomic Bomb
    "What moral dilemmas faced scientists during the Manhattan Project, and how are they relevant today?",
    "How did the development of nuclear weapons transform the relationship between science and government?"
] 

# List of personalities
personalities = [
    "Chanakya", "Sun Tzu", "Machiavelli", "Leonardo da Vinci", "Socrates", "Plato", "Aristotle",
    "Confucius", "Marcus Aurelius", "Friedrich Nietzsche", "Carl Jung", "Sigmund Freud",
    "Winston Churchill", "Abraham Lincoln", "Mahatma Gandhi", "Martin Luther King Jr.", "Nelson Mandela",
    "Albert Einstein", "Isaac Newton", "Marie Curie", "Stephen Hawking", "Richard Feynman", "Nikola Tesla",
    "Galileo Galilei", "James Clerk Maxwell", "Charles Darwin",
    "Alan Turing", "Claude Shannon", "Ada Lovelace", "Grace Hopper", "Tim Berners-Lee",
    "Linus Torvalds", "Guido van Rossum", "Dennis Ritchie",
    "Bill Gates", "Steve Jobs", "Elon Musk", "Jeff Bezos", "Satya Nadella", "Tim Cook",
    "Lisa Su", "Larry Page", "Sergey Brin", "Mark Zuckerberg", "Jensen Huang",
    "Gordon Moore", "Robert Noyce", "Andy Grove"
]


def parse_lesson_to_dict(lesson_text: str, topic: str = "") -> Dict[str, Any]:
    """
    Parse lesson text with [KEY INSIGHT], [HISTORICAL], [APPLICATION] markers
    into a structured dictionary.
    """
    result = {
        "topic": topic,
        "key_insight": "",
        "historical": "",
        "application": "",
        "raw_text": lesson_text,
    }

    if not lesson_text:
        return result

    key_insight_match = re.search(
        r"\[KEY INSIGHT\]\s*(.*?)(?=\[HISTORICAL\]|\[APPLICATION\]|$)",
        lesson_text,
        re.DOTALL | re.IGNORECASE,
    )
    historical_match = re.search(
        r"\[HISTORICAL\]\s*(.*?)(?=\[KEY INSIGHT\]|\[APPLICATION\]|$)",
        lesson_text,
        re.DOTALL | re.IGNORECASE,
    )
    application_match = re.search(
        r"\[APPLICATION\]\s*(.*?)(?=\[KEY INSIGHT\]|\[HISTORICAL\]|$)",
        lesson_text,
        re.DOTALL | re.IGNORECASE,
    )

    if key_insight_match and key_insight_match.group(1).strip():
        result["key_insight"] = key_insight_match.group(1).strip()
    else:
        first_marker_pos = len(lesson_text)
        for marker in [r"\[HISTORICAL\]", r"\[APPLICATION\]"]:
            match = re.search(marker, lesson_text, re.IGNORECASE)
            if match and match.start() < first_marker_pos:
                first_marker_pos = match.start()
        if first_marker_pos > 0:
            pre_text = lesson_text[:first_marker_pos].strip()
            if len(pre_text) > 30:
                result["key_insight"] = pre_text
        if not result["key_insight"]:
            result["key_insight"] = (
                "Timeless principles reveal themselves through historical patterns - "
                "understanding the past illuminates the path forward."
            )

    if historical_match and historical_match.group(1).strip():
        result["historical"] = historical_match.group(1).strip()

    if application_match and application_match.group(1).strip():
        result["application"] = application_match.group(1).strip()

    return result


def get_random_personality():
    used_personalities_file = "used_personalities.txt"

    if os.path.exists(used_personalities_file):
        with open(used_personalities_file, "r") as file:
            used_personalities = file.read().splitlines()
    else:
        used_personalities = []

    unused_personalities = list(set(personalities) - set(used_personalities))

    if not unused_personalities:
        unused_personalities = personalities.copy()
        used_personalities = []

    personality = random.choice(unused_personalities)
    used_personalities.append(personality)

    with open(used_personalities_file, "w") as file:
        for used_personality in used_personalities:
            file.write(f"{used_personality}\n")

    return personality


def get_random_topic():
    used_topics_file = "used_topics.txt"

    if os.path.exists(used_topics_file):
        with open(used_topics_file, "r") as file:
            used_topics = file.read().splitlines()
    else:
        used_topics = []

    unused_topics = list(set(topics) - set(used_topics))

    if not unused_topics:
        unused_topics = topics.copy()
        used_topics = []

    topic = random.choice(unused_topics)
    used_topics.append(topic)

    with open(used_topics_file, "w") as file:
        for used_topic in used_topics:
            file.write(f"{used_topic}\n")

    return topic


def get_random_lesson(llm_provider: str) -> Tuple[str, str]:
    """
    Generate a comprehensive daily lesson using LLM's knowledge.
    Returns (topic, lesson_text).
    """
    topic = get_random_topic()
    prompt = f"""Topic: {topic}

YOU MUST START YOUR RESPONSE WITH "[KEY INSIGHT]" - this is mandatory.

Generate wisdom in EXACTLY this structured format:

[KEY INSIGHT]
Write 1-2 powerful sentences capturing the core wisdom about this topic.

[HISTORICAL]
Share ONE brief historical anecdote (2-3 sentences) that illustrates this principle.

[APPLICATION]
Write 1-2 sentences on how this applies to modern engineering or leadership.

CRITICAL RULES:
1. Your FIRST line MUST be "[KEY INSIGHT]" followed by your insight
2. Then "[HISTORICAL]" followed by your historical example
3. Then "[APPLICATION]" followed by your application
4. Keep each section brief - this is for a newsletter
5. Do NOT skip any section - all three are required"""

    lesson_learned = generate_lesson_response(prompt, llm_provider)
    return topic, lesson_learned


def generate_lesson_response(user_message: str, llm_provider: str) -> str:
    """
    Generate a comprehensive lesson/learning content with historical context.
    """
    logger.info("Generating lesson for topic: %s...", user_message[:100])
    client = get_client(provider=llm_provider, model_tier="fast")

    syspromptmessage = """You are EDITH, an expert teacher helping Dinesh master timeless principles of success.

MANDATORY FORMAT - YOUR RESPONSE MUST LOOK EXACTLY LIKE THIS:
[KEY INSIGHT]
<your key insight here - 1-2 sentences>

[HISTORICAL]
<your historical example here - 2-3 sentences>

[APPLICATION]
<your application here - 1-2 sentences>

RULES:
1. START with "[KEY INSIGHT]" on the very first line - no exceptions
2. Include ALL THREE sections in order: KEY INSIGHT, HISTORICAL, APPLICATION
3. Keep each section brief (1-3 sentences)
4. NO introductions, NO meta-commentary - just the formatted content

Context: Dinesh is a Senior Analog Circuit Design Engineer who values first principles thinking."""

    conversation = [
        {"role": "system", "content": syspromptmessage},
        {"role": "user", "content": user_message},
    ]

    try:
        message = client.chat_completion(
            messages=conversation,
            max_tokens=4000,
            temperature=0.65,
        )

        logger.debug(
            "Raw LLM response for lesson (first 300 chars): %s",
            message[:300] if message else "EMPTY",
        )
        logger.debug("Full raw response length: %s", len(message) if message else 0)

        if not message or len(message.strip()) == 0:
            logger.warning("LLM returned empty response for lesson generation")
            return _get_fallback_lesson()

        cleaned = message.strip()

        reasoning_patterns = [
            "the user wants", "user wants", "user says", "user is asking",
            "we need to", "we need", "we must", "let me provide", "let me",
            "here's", "here is", "i'll provide", "i'll", "i will",
            "task:", "topic:", "provide a", "generate", "create a lesson",
            "write a lesson", "this lesson", "this response", "my response",
            "to answer this", "in response", "based on"
        ]

        lines = cleaned.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(
                pattern in line.lower() for pattern in reasoning_patterns
            ):
                start_idx = i
                break
        cleaned = "\n".join(lines[start_idx:]).strip()

        if cleaned.lower().startswith("assistant:"):
            cleaned = cleaned[len("assistant:") :].strip()
        if cleaned.lower().startswith("assistant"):
            cleaned = cleaned[len("assistant") :].strip()
        if cleaned.lower().startswith("edith"):
            cleaned = cleaned[len("edith") :].strip()

        for marker in ["[KEY INSIGHT]", "[HISTORICAL]", "[APPLICATION]"]:
            if marker.lower() not in cleaned.lower():
                logger.warning("Missing required marker (%s), using fallback", marker)
                return _get_fallback_lesson()

        reasoning_patterns += [
            "reasoning process",
            "let's think step by step",
            "analysis",
            "let me think",
            "i will",
        ]
        for pattern in reasoning_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"(?m)^\s*(analysis|reasoning):\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(
            r"(?m)^\s*(thinking|thought process):\s*", "", cleaned, flags=re.I
        )
        cleaned = cleaned.strip()

        if cleaned.lower().startswith(("[key insight", "key insight")) is False:
            if "[KEY INSIGHT]" not in cleaned:
                cleaned = "[KEY INSIGHT]\n" + cleaned

        first_para = cleaned.split("\n\n")[0] if "\n\n" in cleaned else cleaned[:250]
        if any(pattern in first_para.lower() for pattern in reasoning_patterns):
            logger.debug("Found reasoning patterns in first paragraph, using second paragraph")
            paragraphs = cleaned.split("\n\n")
            if len(paragraphs) > 1:
                cleaned = "\n\n".join(paragraphs[1:]).strip()

        if cleaned and len(cleaned) > 200:
            logger.info("Successfully generated lesson (%s chars)", len(cleaned))
            return cleaned

        logger.warning(
            "Generated lesson too short (%s chars), using fallback",
            len(cleaned) if cleaned else 0,
        )
        return _get_fallback_lesson()

    except Exception as e:
        logger.error("Error generating lesson: %s", e, exc_info=True)
        return _get_fallback_lesson()


def _get_fallback_lesson():
    """Return a fallback lesson when generation fails"""
    fallback = """The pursuit of mastery requires understanding that excellence is not a destination but a continuous journey. Ancient philosophers like Aristotle understood this, coining the term "eudaimonia" to describe the flourishing that comes from living up to one's potential through disciplined practice.

Consider the example of Leonardo da Vinci, who kept detailed notebooks throughout his life, documenting not just his artistic techniques but his observations of nature, engineering concepts, and philosophical musings. This habit of systematic learning and documentation allowed him to make connections across domains that others missed.

For modern engineers and leaders, this translates to three practical principles: First, maintain a learning system - whether notebooks, digital tools, or structured reflection time. Second, seek cross-domain knowledge, as innovation often happens at the intersection of fields. Third, embrace deliberate practice over mere repetition, focusing on the areas where improvement is most needed.

The historical parallel to the Roman aqueduct engineers is instructive: they built systems that lasted millennia not through revolutionary innovation alone, but through meticulous attention to fundamentals, redundancy in design, and deep understanding of the materials and forces they worked with."""

    logger.info("Using fallback lesson content")
    return fallback
