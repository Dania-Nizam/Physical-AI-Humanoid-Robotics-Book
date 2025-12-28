import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <Heading as="h1" className={clsx('hero__title', styles.glowingTitle)}>
            {siteConfig.title}
          </Heading>
          <p className={clsx('hero__subtitle', styles.subtitle)}>
            {siteConfig.tagline}
          </p>
          <div className={styles.buttons}>
            <Link
              className={clsx('button button--secondary button--lg futuristic-glow', styles.heroButton)}
              to="/docs/intro">
              Get Started 🚀
            </Link>
            <Link
              className={clsx('button button--outline button--secondary button--lg futuristic-glow', styles.heroButton)}
              to="/docs/module-1-robotic-nervous-system/introduction-to-ros2">
              Explore Modules
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout title="Physical AI & Robotics" description="Mastering Humanoid Systems">
      <main>
        <HomepageHeader />
        {/* Yahan aur kuch add nahi karna agar sirf hero chahiye */}
      </main>
    </Layout>
  );
}