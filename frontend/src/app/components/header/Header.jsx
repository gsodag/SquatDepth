'use client'
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import logo from '@/app/images/SquatAIlogoWhiteToSharpen.png';

const Header = () => {
  const router = useRouter();

  const handleClick = (label) => {
    switch(label){
      case "SquatAI":
        router.push('/pages/squatAI');
        break;
      case "Load your Barbell":
        router.push('/pages/barbell');
        break;
      case "DOTS Calculator":
        router.push('/pages/calculator');
        break;
      case "1RM Calculator":
        router.push('/pages/repMax');
        break;
      default:
        break;
    }
  };

  const buttons = [
    "SquatAI",
    "Load your Barbell",
    "DOTS Calculator",
    "1RM Calculator",
  ];

  return (
    <header className="w-full bg-zinc-950 shadow-xl z-50">
      <nav className="max-w-7xl mx-auto flex items-center justify-between px-4 py-3">
        
        <button
          className="flex items-center gap-3 group"
          onClick={() => router.push('/pages/squatAI')}
        >
          <span className="relative flex items-center">
            <Image
              src={logo}
              alt="SQUATAI"
              className="h-12 w-auto"
              priority
            />
          </span>
        </button>
        
        <ul className="flex gap-2 md:gap-4 items-center">
          {buttons.map((label) => (
            <li key={label}>
              <button
                onClick={() => handleClick(label)}
                className="px-4 py-2 rounded-xl font-bold uppercase tracking-wide transition duration-300 ease-in-out
                  bg-zinc-800 text-zinc-200 hover:bg-red-600 hover:text-white shadow-lg
                  border border-zinc-700 hover:border-red-500 focus:outline-none focus:ring-4 focus:ring-red-700/50 transform hover:scale-105"
              >
                {label}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <div className="w-full h-1.5 bg-red-600 shadow-xl"></div>
    </header>
  );
};

export default Header;