'use client'
import { useRouter } from 'next/navigation';
import Image from 'next/image';
//import logo from '@/app/images/SquatAIlogo.png';
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
    <header className="w-full bg-zinc-950 border-b-4 border-red-700 shadow-lg z-50">
      <nav className="max-w-7xl mx-auto flex items-center justify-between px-4 py-2">
        <button
          className="flex items-center gap-3 group"
          onClick={() => router.push('/')}
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
                className="px-4 py-2 rounded-xl font-bold uppercase tracking-wide transition
                  bg-zinc-900 text-zinc-100 hover:bg-red-600 hover:text-white shadow-md
                  border-2 border-zinc-800 hover:border-red-500 focus:outline-none focus:ring-2 focus:ring-red-700"
              >
                {label}
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
};

export default Header;