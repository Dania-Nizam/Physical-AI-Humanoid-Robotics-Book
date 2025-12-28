import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical AI Concepts',
    // Aap static/img mein apni images dal kar path change kar sakti hain
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default, 
    description: (
      <>
        Learn the core components of Physical AI, from high-fidelity 
        simulations to real-time sensor processing and cognitive planning.
      </>
    ),
  },
  {
    title: 'ROS 2 Architecture',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Master the industry-standard Robot Operating System (ROS 2) to 
        build scalable, modular, and robust robotic applications.
      </>
    ),
  },
  {
    title: 'Isaac Sim & NVIDIA Omniverse',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Harness the power of NVIDIA Isaac Sim for synthetic data generation, 
        physics-accurate environments, and advanced robot training.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
