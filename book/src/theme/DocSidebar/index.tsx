import React from 'react';
import DocSidebarItems from '@theme/DocSidebarItems';
import clsx from 'clsx';
import styles from './styles.module.css';

export default function DocSidebar(props: any) {
  const {sidebar, path} = props;

  // Is console ko check karein browser ke inspect element mein
  console.log('Sidebar Data:', sidebar);

  if (!sidebar ) {
    // Agar sidebar undefined hai ya items undefined hai, kuch render na karo
    return null;
  }

  const items = Array.isArray(sidebar) ? sidebar : (sidebar.items ?? []);

  return (
    <div className={clsx('sidebar-container', styles.sidebarContainer)}>
      <nav
        aria-label="Sidebar navigation"
        className={clsx('menu thin-scrollbar', styles.sidebarMenu)}>
        <ul className={clsx('menu__list', styles.sidebarList, styles.sidebarItems)}>
          <DocSidebarItems items={items} activePath={path} level={1} />
        </ul>
      </nav>
    </div>
  );
}
